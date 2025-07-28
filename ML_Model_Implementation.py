import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import urllib
import xgboost as xgb
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

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
    print("📊 Loading data from database...")
    df_requests = pd.read_sql("SELECT * FROM ride_requests", con=engine)
    df_completion = pd.read_sql("SELECT * FROM ride_completion_delay", con=engine)
    df_acceptance = pd.read_sql("SELECT * FROM ride_acceptance_delay", con=engine)
    df_matches = pd.read_sql("SELECT * FROM ride_matches", con=engine)
    print(f"✅ Data loaded successfully!")
    print(f"   - Requests: {len(df_requests)} rows")
    print(f"   - Completion: {len(df_completion)} rows")
    print(f"   - Acceptance: {len(df_acceptance)} rows")
    print(f"   - Matches: {len(df_matches)} rows")
except Exception as e:
    print(f"❌ Error loading data from database: {e}")
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

print("\n🔧 Creating feature dataset...")

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

print(f"✅ Features dataset created with {len(df_features)} rows")

# Sort by matched_at time
df_features_sorted = df_features.sort_values(by='matched_at', ascending=True)

def compute_surge_hourly_stats(df_features, df_requests, df_acceptance, df_completion):
    """
    Compute hourly statistics using the actual data range for surge pricing analysis.
    This replaces the previous week lookup with same-week hourly aggregations.
    """
    print("📈 Computing hourly surge statistics...")

    # Get the actual data range
    data_start = df_features['matched_at'].min()
    data_end = df_features['matched_at'].max()

    print(f"   Data range: {data_start} to {data_end}")

    # Create hourly bins for the entire dataset
    df_features['hour_bin'] = df_features['matched_at'].dt.floor('H')
    df_requests['hour_bin'] = df_requests['request_time'].dt.floor('H')
    df_acceptance['hour_bin'] = df_acceptance['request_time'].dt.floor('H')
    df_completion['hour_bin'] = df_completion['pickup_time'].dt.floor('H')

    # Pre-compute hourly aggregations for efficiency
    print("   Pre-computing hourly aggregations...")

    # Request counts per hour
    hourly_requests = df_requests.groupby('hour_bin').size().reset_index(name='request_count')

    # Acceptance time stats per hour
    hourly_acceptance = df_acceptance.groupby('hour_bin')['delay_seconds'].agg([
        ('avg_accept_time', 'mean'),
        ('std_accept_time', 'std')
    ]).reset_index()
    hourly_acceptance['std_accept_time'] = hourly_acceptance['std_accept_time'].fillna(0)

    # Ride duration stats per hour
    hourly_completion = df_completion.groupby('hour_bin')['duration_seconds'].agg([
        ('avg_ride_duration', 'mean'),
        ('std_ride_duration', 'std')
    ]).reset_index()
    hourly_completion['std_ride_duration'] = hourly_completion['std_ride_duration'].fillna(0)

    # Merge all hourly stats
    hourly_stats = hourly_requests.merge(hourly_acceptance, on='hour_bin', how='outer')
    hourly_stats = hourly_stats.merge(hourly_completion, on='hour_bin', how='outer')

    # Fill missing values
    hourly_stats = hourly_stats.fillna(0)

    print(f"   Generated hourly stats for {len(hourly_stats)} hours")

    # Compute overall statistics for z-score calculations
    overall_request_mean = hourly_stats['request_count'].mean()
    overall_request_std = hourly_stats['request_count'].std()
    overall_accept_mean = hourly_stats['avg_accept_time'].mean()
    overall_accept_std = hourly_stats['avg_accept_time'].std()

    # Prevent division by zero
    overall_request_std = max(overall_request_std, 0.001)
    overall_accept_std = max(overall_accept_std, 0.001)

    print(f"   Overall request stats - Mean: {overall_request_mean:.2f}, Std: {overall_request_std:.2f}")
    print(f"   Overall accept time stats - Mean: {overall_accept_mean:.2f}, Std: {overall_accept_std:.2f}")

    # Now assign statistics to each ride
    stats = []

    for idx, row in df_features.iterrows():
        if idx % 500 == 0:
            print(f"   Processing ride {idx}/{len(df_features)}")

        ride_hour_bin = row['hour_bin']

        # Find the corresponding hourly stats
        hour_stats = hourly_stats[hourly_stats['hour_bin'] == ride_hour_bin]

        if len(hour_stats) > 0:
            hour_stat = hour_stats.iloc[0]
            request_count = hour_stat['request_count']
            avg_accept_time = hour_stat['avg_accept_time']
            std_accept_time = max(hour_stat['std_accept_time'], 0.001)
            avg_ride_duration = hour_stat['avg_ride_duration']
            std_ride_duration = max(hour_stat['std_ride_duration'], 0.001)
        else:
            # Use overall averages if no specific hour data
            request_count = overall_request_mean
            avg_accept_time = overall_accept_mean
            std_accept_time = overall_accept_std
            avg_ride_duration = df_completion['duration_seconds'].mean()
            std_ride_duration = max(df_completion['duration_seconds'].std(), 0.001)

        # Compute z-scores for surge indicators
        z_request_count = (request_count - overall_request_mean) / overall_request_std
        z_accept_time = (row['current_accept_time'] - avg_accept_time) / std_accept_time

        stats.append({
            'ride_id': row['ride_id'],
            'request_count': request_count,
            'avg_accept_time': avg_accept_time,
            'avg_ride_duration': avg_ride_duration,
            'std_accept_time': std_accept_time,
            'std_ride_duration': std_ride_duration,
            'z_request_count': z_request_count,
            'z_accept_time': z_accept_time,
            'hour_bin': ride_hour_bin
        })

    stats_df = pd.DataFrame(stats)
    print(f" Computed statistics for {len(stats_df)} rides")

    return stats_df

# Compute surge-based statistics
stats_df = compute_surge_hourly_stats(df_features_sorted, df_requests, df_acceptance, df_completion)

# Merge with features
df_merged = df_features_sorted.merge(stats_df, on='ride_id', how='left')

# Remove any rows with missing critical data
df_merged = df_merged.dropna(subset=['price', 'distance_km', 'current_accept_time'])

print(f" Statistics computed and merged. Final dataset: {len(df_merged)} rows")

# Define features for modeling (focusing on surge pricing factors)
feature_cols = [
    'distance_km',           # Base distance factor
    'current_accept_time',   # Current ride's acceptance delay
    'request_count',         # Demand indicator (rides requested in this hour)
    'avg_accept_time',       # Average acceptance time for this hour
    'avg_ride_duration',     # Average ride duration for this hour
    'z_request_count',       # Normalized demand surge indicator
    'z_accept_time'          # Normalized supply shortage indicator
]

# Check for missing columns
missing_cols = [col for col in feature_cols if col not in df_merged.columns]
if missing_cols:
    print(f" Error: Missing required columns: {missing_cols}")
    exit()

# Prepare feature matrix and target
X = df_merged[feature_cols]
y = df_merged['price']

print(f"\n SURGE PRICING MODEL TRAINING:")
print(f"   Target variable: PRICE (ZAR)")
print(f"   Features: {feature_cols}")
print(f"   Feature matrix shape: {X.shape}")
print(f"   Price range: R{y.min():.2f} - R{y.max():.2f}")
print(f"   Average price: R{y.mean():.2f}")
print(f"   Price std dev: R{y.std():.2f}")

# Check for data quality
print(f"\n Data Quality Check:")
print(f"   Non-zero prices: {(y > 0).sum()}/{len(y)}")
print(f"   Average distance: {X['distance_km'].mean():.3f} km")
print(f"   Average requests per hour: {X['request_count'].mean():.1f}")
print(f"   Average accept time: {X['avg_accept_time'].mean():.1f} seconds")

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data for the Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize regression models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    'NeuralNetwork': Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
}

model_results = {}
performance_results = {}

print("\n Training surge pricing models...")

# Fit models and evaluate performance
for name, mdl in models.items():
    print(f"\n--- Training {name} ---")

    if name == 'NeuralNetwork':
        mdl.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = mdl.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        y_train_pred = mdl.predict(X_train_scaled).flatten()
        y_test_pred = mdl.predict(X_test_scaled).flatten()
    else:
        mdl.fit(X_train, y_train)
        y_train_pred = mdl.predict(X_train)
        y_test_pred = mdl.predict(X_test)

    # Calculate performance metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Store results
    if name in ['RandomForest', 'XGBoost']:
        feature_importances = dict(zip(feature_cols, mdl.feature_importances_))
        model_results[name] = feature_importances
        print(f"   Feature Importances: {feature_importances}")
    
    performance_results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }

    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test MAE: R{test_mae:.2f}")
    print(f"   Test RMSE: R{test_rmse:.2f}")

# Print detailed Random Forest results
print("\n" + "="*70)
print(" DETAILED RANDOM FOREST REGRESSION RESULTS:")
print("="*70)

random_forest_perf = performance_results['RandomForest']
feature_importances = model_results['RandomForest']

print(f" Model Performance:")
print(f"   Training R²: {random_forest_perf['train_r2']:.4f}")
print(f"   Test R²: {random_forest_perf['test_r2']:.4f}")
print(f"   Test MAE: R{random_forest_perf['test_mae']:.2f}")
print(f"   Test RMSE: R{random_forest_perf['test_rmse']:.2f}")

print(f"\n🔍 Feature Importance Analysis:")
for feature, importance in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
    print(f"   {feature:20}: {importance:.4f}")

# Print detailed XGBoost results
print("\n" + "="*70)
print(" DETAILED XGBOOST REGRESSION RESULTS:")
print("="*70)

xgboost_perf = performance_results['XGBoost']
feature_importances_xgb = model_results['XGBoost']

print(f" Model Performance:")
print(f"   Training R²: {xgboost_perf['train_r2']:.4f}")
print(f"   Test R²: {xgboost_perf['test_r2']:.4f}")
print(f"   Test MAE: R{xgboost_perf['test_mae']:.2f}")
print(f"   Test RMSE: R{xgboost_perf['test_rmse']:.2f}")

print(f"\n🔍 Feature Importance Analysis:")
for feature, importance in sorted(feature_importances_xgb.items(), key=lambda item: item[1], reverse=True):
    print(f"   {feature:20}: {importance:.4f}")

# Print detailed Neural Network results
print("\n" + "="*70)
print(" DETAILED NEURAL NETWORK REGRESSION RESULTS:")
print("="*70)
nn_perf = performance_results['NeuralNetwork']
print(f" Model Performance:")
print(f"   Training R²: {nn_perf['train_r2']:.4f}")
print(f"   Test R²: {nn_perf['test_r2']:.4f}")
print(f"   Test MAE: R{nn_perf['test_mae']:.2f}")
print(f"   Test RMSE: R{nn_perf['test_rmse']:.2f}")

# Model comparison
print(f"\n📊 MODEL COMPARISON (Test R²):")
for name, perf in performance_results.items():
    print(f"   {name:12}: {perf['test_r2']:.4f} (MAE: R{perf['test_mae']:.2f})")

best_model_name = max(performance_results.keys(), key=lambda x: performance_results[x]['test_r2'])
print(f"   🏆 Best Model: {best_model_name}")

# Export the best model and the scaler to a PKL file
best_model_obj = models[best_model_name]

# For a Neural Network, you'd save the Keras model separately, but for consistency, we'll
# save the scikit-learn models directly. The user can adapt if NN is best.
if best_model_name != 'NeuralNetwork':
    model_filename = f'best_pricing_model.pkl'
    joblib.dump(best_model_obj, model_filename)
    print(f"\n💾 Best model '{best_model_name}' saved to {model_filename}")

    # Also save the scaler if it was a Neural Network or another scaled model
    if best_model_name == 'NeuralNetwork':
         joblib.dump(scaler, 'scaler_for_nn.pkl')
         print("💾 Scaler saved to scaler_for_nn.pkl")

# Sample prediction with surge analysis
if len(X_test) > 0:
    sample_idx = 0
    sample_features = X_test.iloc[sample_idx]
    
    # Use the best model for prediction
    if best_model_name == 'NeuralNetwork':
        sample_features_scaled = scaler.transform([sample_features])
        predicted_price = best_model_obj.predict(sample_features_scaled).flatten()[0]
    else:
        predicted_price = best_model_obj.predict([sample_features])[0]
    
    actual_price = y_test.iloc[sample_idx]
    
    print(f"\n🔍 SAMPLE SURGE PRICING PREDICTION using the {best_model_name} model:")
    print(f"   💰 Actual Price: R{actual_price:.2f}")
    print(f"   🎯 Predicted Price: R{predicted_price:.2f}")
    print(f"   📊 Prediction Error: R{abs(actual_price - predicted_price):.2f}")

    print(f"\n   📋 Sample Features:")
    for feature, value in sample_features.items():
        print(f"      {feature}: {value:.4f}")

print(f"\n🎉 Surge pricing model training completed successfully!")
print(f"📈 Best model achieved {performance_results[best_model_name]['test_r2']:.1%} accuracy on test data")