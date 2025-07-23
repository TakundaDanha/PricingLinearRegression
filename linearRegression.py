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

# Construct ODBC connection string
odbc_str = (
    f"DRIVER=ODBC Driver 17 for SQL Server;"
    f"SERVER={DB_HOST},{DB_PORT};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    "TrustServerCertificate=yes;"
)

# Create database connection
connection_uri = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_str)}"
engine = create_engine(connection_uri)

# Load datasets from database
try:
    df_features = pd.read_sql("SELECT * FROM ride_pricing_features", con=engine)
    df_requests = pd.read_sql("SELECT * FROM ride_requests", con=engine)
    df_completion = pd.read_sql("SELECT * FROM ride_completion_delay", con=engine)
    df_acceptance = pd.read_sql("SELECT * FROM ride_acceptance_delay", con=engine)
except Exception as e:
    print(f"Error loading data from database: {e}")
    exit()

# Clean column names
for df in [df_features, df_requests, df_completion, df_acceptance]:
    df.columns = df.columns.str.strip().str.replace('""', '').str.replace('"', '')

# Clean data values (fix commas in numeric columns)
for df in [df_features, df_requests, df_completion, df_acceptance]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.')
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

# Debug: Inspect datasets
print("ride_pricing_features columns:", df_features.columns.tolist())
print("ride_pricing_features shape:", df_features.shape)
print("\nride_pricing_features head:")
print(df_features.head())

# Sort requests by request_time
df_requests_sorted = df_requests.sort_values(by='request_time', ascending=False)

# Convert datetime columns
for df in [df_features, df_requests_sorted, df_acceptance, df_completion]:
    if 'request_time' in df.columns:
        df['request_time'] = pd.to_datetime(df['request_time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    if 'pickup_time' in df.columns:
        df['pickup_time'] = pd.to_datetime(df['pickup_time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    if 'accepted_time' in df.columns:
        df['accepted_time'] = pd.to_datetime(df['accepted_time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    if 'completed_at' in df.columns:
        df['completed_at'] = pd.to_datetime(df['completed_at'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

# Function to compute hourly statistics
def compute_hourly_stats(df_features, df_requests, df_acceptance, df_completion):
    stats = []
    base_date = (df_requests_sorted['request_time'].iloc[0].floor('D')
                 if 'request_time' in df_requests.columns and not df_requests['request_time'].isna().all()
                 else pd.to_datetime('2025-01-01'))

    for _, row in df_features.iterrows():
        ride_id = row['ride_id']
        hour = row['request_hour']
        lag_hour = hour - 1
        lead_hour = hour + 1

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
        df_acceptance['accept_time_diff'] = (df_acceptance['accepted_time'] - df_acceptance['request_time']).dt.total_seconds()
        accept_times = df_acceptance[mask_acceptance]['accept_time_diff']
        avg_accept_time = accept_times.mean() if not accept_times.empty else 0
        stdev_accept_time = accept_times.std() if len(accept_times) > 1 else 0

        # Compute ride duration statistics
        df_completion['ride_duration'] = (df_completion['completed_at'] - df_completion['pickup_time']).dt.total_seconds()
        ride_durations = df_completion[mask_completion]['ride_duration']
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

# Debug: Inspect merged dataset
print("\nMerged dataset columns:", df_merged.columns.tolist())
print("Merged dataset shape:", df_merged.shape)
print("\nMerged dataset head:")
print(df_merged.head())

# Define features for modeling
feature_cols = [
    'distance_km', 'current_accept_time', 'request_count', 'avg_accept_time',
    'avg_ride_duration', 'z_request_count', 'z_accept_time'
]
missing_cols = [col for col in feature_cols if col not in df_merged.columns]

if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    exit()

# Prepare feature matrix and target
X = df_merged[feature_cols]
y = df_merged['duration_min']

print(f"\nFeatures being used: {feature_cols}")
print(f"Feature matrix shape: {X.shape}")

# Initialize regression models
models = {
    'Linear': LinearRegression(),
    'Ridge': RidgeCV(alphas=[0.1, 1.0, 10.0]),
    'Lasso': LassoCV(alphas=[0.01, 0.1, 1.0]),
    'ElasticNet': ElasticNetCV(alphas=[0.01, 0.1, 1.0], l1_ratio=[0.2, 0.5, 0.8])
}

model_results = {}

# Fit models and print coefficients
for name, mdl in models.items():
    mdl.fit(X, y)
    coefs = dict(zip(feature_cols, mdl.coef_))
    model_results[name] = coefs
    print(f"\n{name} Regression Coefficients:")
    for feature, coef in coefs.items():
        print(f"{feature}: {round(coef, 4)}")

# Print linear regression results
print("\n" + "="*50)
print("MODEL RESULTS (Predicting duration_min):")
print("="*50)
print("Intercept:", round(models['Linear'].intercept_, 4))
for feature, coef in zip(feature_cols, models['Linear'].coef_):
    print(f"Coefficient for {feature}: {round(coef, 4)}")

# Suggest coefficients for pricing algorithm
print("\n" + "="*50)
print("SUGGESTED PRICING ALGORITHM COEFFICIENTS FOR STORED PROCEDURE:")
print("="*50)

# Feature to pricing variable mapping
feature_to_pricing = {
    'distance_km': 'baseRate',
    'request_count': 'coeffRequests',
    'avg_accept_time': 'coeffAcceptTime',
    'avg_ride_duration': 'coeffRideDuration',
    'z_request_count': 'stdDevFactor',
    'z_accept_time': 'stdDevFactor'
}

# Normalize coefficients for Linear model
included_coefs = [abs(coef) for feature, coef in zip(feature_cols, models['Linear'].coef_) if feature != 'distance_km']
included_sum = sum(included_coefs) or 1
normalized_coefs = [
    abs(coef) / included_sum if feature != 'distance_km' else coef
    for feature, coef in zip(feature_cols, models['Linear'].coef_)
]

# Print coefficients for all models
for name, coefs in model_results.items():
    print(f"\n--- {name} Pricing Coefficients ---")
    included = [abs(c) for f, c in coefs.items() if f != 'distance_km']
    total = sum(included) or 1
    std_dev_features = ['z_request_count', 'z_accept_time']
    std_dev_sum = sum(abs(coefs[f]) for f in std_dev_features) / len(std_dev_features)

    for f in feature_cols:
        key = feature_to_pricing.get(f)
        if key == 'baseRate':
            print(f"DECLARE @{key} FLOAT = 7.85;")
        elif key == 'stdDevFactor':
            continue
        else:
            norm = abs(coefs[f]) / total if f != 'distance_km' else coefs[f]
            print(f"DECLARE @{key} FLOAT = {round(norm, 4)};")
    print(f"DECLARE @stdDevFactor FLOAT = {round(std_dev_sum / total, 4)};")

# Print normalized DECLARE statements
print("\nDECLARE statements for CalculateDynamicPrice (normalized):")
for feature, norm_coef in zip(feature_cols, normalized_coefs):
    pricing_key = feature_to_pricing.get(feature)
    if pricing_key == 'baseRate':
        print(f"DECLARE @{pricing_key} FLOAT = 7.85;")
    elif pricing_key == 'stdDevFactor':
        continue
    else:
        print(f"DECLARE @{pricing_key} FLOAT = {round(norm_coef, 4)};")

# Compute and print stdDevFactor
std_dev_features = ['z_request_count', 'z_accept_time']
std_dev_coef_sum = sum(
    abs(coef) / included_sum
    for feature, coef in zip(feature_cols, models['Linear'].coef_)
    if feature in std_dev_features
) / len(std_dev_features)
print(f"DECLARE @stdDevFactor FLOAT = {round(std_dev_coef_sum, 4)};")