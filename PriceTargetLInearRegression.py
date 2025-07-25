import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from datetime import timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import urllib

# Load environment variables from .env file
load_dotenv()

# Database connection configuration
DB_HOST = os.getenv('SQLSERVER_HOST')
DB_PORT = os.getenv('SQLSERVER_PORT', '1433')
DB_NAME = os.getenv('SQLSERVER_DB')
DB_USER = os.getenv('SQLSERVER_USER')
DB_PASSWORD = os.getenv('SQLSERVER_PASSWORD')

# Construct ODBC connection string with timeout settings
odbc_str = (
    f"DRIVER=ODBC Driver 17 for SQL Server;"
    f"SERVER={DB_HOST},{DB_PORT};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    "TrustServerCertificate=yes;"
    "Connection Timeout=30;"
    "Command Timeout=60;"
)

# Create database connection
connection_uri = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_str)}"
engine = create_engine(
    connection_uri,
    pool_timeout=30,
    pool_recycle=300,
    pool_pre_ping=True
)

# Load datasets from database
try:
    print(" Loading data from database...")
    df_requests = pd.read_sql("SELECT * FROM ride_requests", con=engine)
    df_completion = pd.read_sql("SELECT * FROM ride_completion_delay", con=engine)
    df_acceptance = pd.read_sql("SELECT * FROM ride_acceptance_delay", con=engine)
    df_matches = pd.read_sql("SELECT * FROM ride_matches", con=engine)
    print(f" Data loaded successfully!")
    print(f"   - Requests: {len(df_requests)} rows")
    print(f"   - Completion: {len(df_completion)} rows") 
    print(f"   - Acceptance: {len(df_acceptance)} rows")
    print(f"   - Matches: {len(df_matches)} rows")
except Exception as e:
    print(f" Error loading data from database: {e}")
    exit()

# Clean column names
for df in [df_requests, df_completion, df_acceptance, df_matches]:
    df.columns = df.columns.str.strip().str.replace('""', '').str.replace('"', '')

# Clean data values (fix commas in numeric columns)
for df in [df_requests, df_completion, df_acceptance, df_matches]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.')
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

# Convert datetime columns
datetime_cols = {
    'request_time': [df_requests, df_acceptance],
    'pickup_time': [df_completion],
    'accepted_time': [df_acceptance],
    'completed_at': [df_completion, df_matches],
    'matched_at': [df_matches],
    'driver_response_at': [df_matches],
    'started_at': [df_matches],
    'arrived_at': [df_matches]
}

for col_name, dataframes in datetime_cols.items():
    for df in dataframes:
        if col_name in df.columns:
            df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

print("\n Creating feature dataset...")

# Create features dataset from ride_matches (which has price data)
df_features = df_matches.copy()

# Add distance calculation (simplified Euclidean distance)
df_features['distance_km'] = np.sqrt(
    (df_features['dropoff_lat'] - df_features['pickup_lat'])**2 + 
    (df_features['dropoff_lng'] - df_features['pickup_lng'])**2
) * 111  # Rough conversion to km

# Add time-based features
df_features['request_hour'] = df_features['matched_at'].dt.hour
df_features['request_day'] = df_features['matched_at'].dt.dayofweek

# Merge with acceptance delay data to get current accept time
df_acceptance_slim = df_acceptance[['ride_id', 'delay_seconds']].rename(columns={'delay_seconds': 'current_accept_time'})
df_features = df_features.merge(df_acceptance_slim, on='ride_id', how='left')

# Fill missing accept times with median
df_features['current_accept_time'] = df_features['current_accept_time'].fillna(df_features['current_accept_time'].median())

print(f" Features dataset created with {len(df_features)} rows")

# Sort by matched_at time
df_features_sorted = df_features.sort_values(by='matched_at', ascending=False)

# Function to compute hourly statistics
def compute_hourly_stats(df_features, df_requests, df_acceptance, df_completion):
    print(" Computing hourly statistics...")
    stats = []
    base_date = (df_features_sorted['matched_at'].iloc[0].floor('D')
                 if not df_features['matched_at'].isna().all()
                 else pd.to_datetime('2025-01-01'))
    
    print(f"   Base date for statistics: {base_date}")

    for idx, row in df_features.iterrows():
        if idx % 100 == 0:
            print(f"   Processing row {idx}/{len(df_features)}")
            
        ride_id = row['ride_id']
        hour = row['request_hour']
        lag_hour = max(0, hour - 1)
        lead_hour = min(23, hour + 1)

        prev_week_start = base_date - timedelta(weeks=1) + timedelta(hours=lag_hour)
        prev_week_end = base_date - timedelta(weeks=1) + timedelta(hours=lead_hour + 1)

        # Filter data for the previous week
        mask_requests = (
            (df_requests['request_time'] >= prev_week_start) &
            (df_requests['request_time'] < prev_week_end) &
            (df_requests['request_time'].dt.hour.between(lag_hour, lead_hour))
        )
        mask_acceptance = (
            (df_acceptance['request_time'] >= prev_week_start) &
            (df_acceptance['request_time'] < prev_week_end) &
            (df_acceptance['request_time'].dt.hour.between(lag_hour, lead_hour))
        )
        mask_completion = (
            (df_completion['pickup_time'] >= prev_week_start) &
            (df_completion['pickup_time'] < prev_week_end) &
            (df_completion['pickup_time'].dt.hour.between(lag_hour, lead_hour))
        )

        # Compute request statistics
        request_count = df_requests[mask_requests].shape[0]
        hourly_counts = df_requests[mask_requests].groupby(df_requests['request_time'].dt.hour).size()
        avg_request_count = hourly_counts.mean() if not hourly_counts.empty else 0
        stdev_request_count = hourly_counts.std() if len(hourly_counts) > 1 else 0

        # Compute acceptance time statistics
        accept_times = df_acceptance[mask_acceptance]['delay_seconds']
        avg_accept_time = accept_times.mean() if not accept_times.empty else 0
        stdev_accept_time = accept_times.std() if len(accept_times) > 1 else 0

        # Compute ride duration statistics
        ride_durations = df_completion[mask_completion]['duration_seconds']
        avg_ride_duration = ride_durations.mean() if not ride_durations.empty else 0
        stdev_ride_duration = ride_durations.std() if len(ride_durations) > 1 else 0

        # Compute z-scores
        z_request_count = (request_count - avg_request_count) / stdev_request_count if stdev_request_count > 0 else 0
        z_accept_time = (row['current_accept_time'] - avg_accept_time) / stdev_accept_time if stdev_accept_time > 0 else 0

        stats.append({
            'ride_id': ride_id,
            'request_count': request_count,
            'avg_request_count': avg_request_count,
            'stdev_request_count': stdev_request_count,
            'avg_accept_time': avg_accept_time,
            'stdev_accept_time': stdev_accept_time,
            'avg_ride_duration': avg_ride_duration,
            'stdev_ride_duration': stdev_ride_duration,
            'z_request_count': z_request_count,
            'z_accept_time': z_accept_time
        })

    return pd.DataFrame(stats)

# Compute statistics
stats_df = compute_hourly_stats(df_features, df_requests, df_acceptance, df_completion)

# Merge with features
df_merged = df_features.merge(stats_df, on='ride_id', how='left')

# Filter rows where not all statistical columns are zero
stat_columns = [
    'request_count', 'avg_request_count', 'stdev_request_count',
    'avg_accept_time', 'stdev_accept_time', 'avg_ride_duration',
    'stdev_ride_duration', 'z_request_count', 'z_accept_time'
]
df_merged = df_merged[df_merged[stat_columns].ne(0).any(axis=1)]

print(f" Statistics computed and merged. Final dataset: {len(df_merged)} rows")

# Define features for modeling
feature_cols = [
    'distance_km', 'current_accept_time', 'request_count', 'avg_accept_time',
    'avg_ride_duration', 'z_request_count', 'z_accept_time'
]

# Check for missing columns
missing_cols = [col for col in feature_cols if col not in df_merged.columns]
if missing_cols:
    print(f" Error: Missing required columns: {missing_cols}")
    exit()

# Prepare feature matrix and target (NOW USING PRICE)
X = df_merged[feature_cols]
y = df_merged['price']  # Changed from 'duration_min' to 'price'

print(f"\n MODEL TRAINING:")
print(f"   Target variable: PRICE (ZAR)")
print(f"   Features: {feature_cols}")
print(f"   Feature matrix shape: {X.shape}")
print(f"   Price range: R{y.min():.2f} - R{y.max():.2f}")
print(f"   Average price: R{y.mean():.2f}")

# Initialize regression models
models = {
    'Linear': LinearRegression(),
    'Ridge': RidgeCV(alphas=[0.1, 1.0, 10.0]),
    'Lasso': LassoCV(alphas=[0.01, 0.1, 1.0]),
    'ElasticNet': ElasticNetCV(alphas=[0.01, 0.1, 1.0], l1_ratio=[0.2, 0.5, 0.8])
}

model_results = {}

print("\n Training models...")
# Fit models and print coefficients
for name, mdl in models.items():
    mdl.fit(X, y)
    coefs = dict(zip(feature_cols, mdl.coef_))
    model_results[name] = coefs
    
    # Calculate R² score
    r2_score = mdl.score(X, y)
    print(f"\n{name} Regression Results:")
    print(f"   R² Score: {r2_score:.4f}")
    print("   Coefficients:")
    for feature, coef in coefs.items():
        print(f"     {feature}: {coef:.4f}")

# Print detailed linear regression results
print("\n" + "="*60)
print("DETAILED LINEAR REGRESSION RESULTS (Predicting PRICE in ZAR):")
print("="*60)
linear_model = models['Linear']
print(f"Intercept (Base Price): R{linear_model.intercept_:.2f}")
print(f"R² Score: {linear_model.score(X, y):.4f}")
print("\nCoefficient Interpretations:")
for feature, coef in zip(feature_cols, linear_model.coef_):
    print(f"  {feature}: R{coef:.4f}")
    if feature == 'distance_km':
        print(f"    → Each additional km adds R{coef:.2f} to the price")
    elif feature == 'current_accept_time':
        print(f"    → Each additional second of accept time {'increases' if coef > 0 else 'decreases'} price by R{abs(coef):.4f}")
    elif feature == 'request_count':
        print(f"    → Each additional request in the area {'increases' if coef > 0 else 'decreases'} price by R{abs(coef):.4f}")

# Suggest coefficients for pricing algorithm
print("\n" + "="*60)
print("SUGGESTED PRICING ALGORITHM COEFFICIENTS FOR STORED PROCEDURE:")
print("="*60)

# Feature to pricing variable mapping
feature_to_pricing = {
    'distance_km': 'baseRate',
    'request_count': 'coeffRequests', 
    'avg_accept_time': 'coeffAcceptTime',
    'avg_ride_duration': 'coeffRideDuration',
    'z_request_count': 'stdDevFactor',
    'z_accept_time': 'stdDevFactor'
}

# Print coefficients for all models
for name, coefs in model_results.items():
    print(f"\n--- {name} Pricing Coefficients ---")
    included = [abs(c) for f, c in coefs.items() if f != 'distance_km']
    total = sum(included) or 1
    std_dev_features = ['z_request_count', 'z_accept_time']
    std_dev_sum = sum(abs(coefs[f]) for f in std_dev_features) / len(std_dev_features) if std_dev_features else 0

    for f in feature_cols:
        key = feature_to_pricing.get(f)
        if key == 'baseRate':
            print(f"DECLARE @{key} FLOAT = {linear_model.intercept_:.2f};  -- Base price from intercept")
        elif key == 'stdDevFactor':
            continue
        else:
            coef_val = coefs[f]
            print(f"DECLARE @{key} FLOAT = {coef_val:.4f};")
    
    if std_dev_sum > 0:
        print(f"DECLARE @stdDevFactor FLOAT = {std_dev_sum:.4f};")

# Print final pricing formula
print(f"\n" + "="*60)
print("PRICING FORMULA:")
print("="*60)
print("Predicted Price = Intercept + (distance_km × coeff) + (accept_time × coeff) + ...")
print(f"Example: R{linear_model.intercept_:.2f}", end="")
for feature, coef in zip(feature_cols, linear_model.coef_):
    sign = "+" if coef >= 0 else ""
    print(f" {sign} ({feature} × {coef:.4f})", end="")
print()

# Sample prediction
if len(X) > 0:
    sample_idx = 0
    sample_features = X.iloc[sample_idx]
    predicted_price = linear_model.predict([sample_features])[0]
    actual_price = y.iloc[sample_idx]
    
    print(f"\n📋 SAMPLE PREDICTION:")
    print(f"   Actual Price: R{actual_price:.2f}")
    print(f"   Predicted Price: R{predicted_price:.2f}")
    print(f"   Difference: R{abs(actual_price - predicted_price):.2f}")
    print("   Sample Features:")
    for feature, value in sample_features.items():
        print(f"     {feature}: {value:.4f}")

print(f"\n Price prediction model training complete!")