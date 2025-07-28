import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import os
from datetime import datetime, timedelta
import urllib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configure Logging ---
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
app.logger.info("Starting Flask application...")

# --- Load Environment Variables for DB ---
load_dotenv()
DB_HOST = os.getenv('SQLSERVER_HOST')
DB_PORT = os.getenv('SQLSERVER_PORT', '1433')
DB_NAME = os.getenv('SQLSERVER_DB')
DB_USER = os.getenv('SQLSERVER_USER')
DB_PASSWORD = os.getenv('SQLSERVER_PASSWORD')

# Log environment variables (mask password for security)
app.logger.info(f"DB_HOST: {DB_HOST}, DB_PORT: {DB_PORT}, DB_NAME: {DB_NAME}, DB_USER: {DB_USER}")

# --- Construct ODBC Connection String ---
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
connection_uri = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_str)}"
try:
    engine = create_engine(connection_uri, pool_timeout=30, pool_recycle=300, pool_pre_ping=True)
    app.logger.info("Database engine created successfully.")
except Exception as e:
    app.logger.error(f"Failed to create database engine: {e}")
    engine = None

# --- Load the trained model ---
try:
    model = joblib.load('best_pricing_model.pkl')
    app.logger.info("✅ Successfully loaded pricing model.")
except FileNotFoundError:
    app.logger.error("❌ Error: 'best_pricing_model.pkl' not found. Please run the training script first.")
    model = None

# --- Helper function to convert NumPy types to Python types ---
def convert_numpy_types(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# --- Helper function to fetch hourly stats from the DB ---
def fetch_hourly_stats(matched_at: datetime):
    """
    Fetches hourly and overall statistics from the database for a given timestamp.
    """
    try:
        # Convert UTC to SAST (UTC+2)
        sast_offset = timedelta(hours=2)
        matched_at_sast = matched_at + sast_offset
        hour_start = matched_at_sast.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        
        # Format timestamps for SQL Server
        hour_start_str = hour_start.strftime('%Y-%m-%d %H:%M:%S')
        hour_end_str = hour_end.strftime('%Y-%m-%d %H:%M:%S')
        app.logger.info(f"Fetching stats for hour_start={hour_start_str}, hour_end={hour_end_str}")
        
        with engine.connect() as connection:
            # Fetch hourly stats
            hourly_requests = pd.read_sql(
                f"SELECT COUNT(ride_id) AS request_count FROM ride_requests "
                f"WHERE request_time >= '{hour_start_str}' AND request_time < '{hour_end_str}'",
                connection
            )
            app.logger.info(f"hourly_requests: {hourly_requests.to_dict()}")
            
            hourly_acceptance = pd.read_sql(
                f"SELECT AVG(delay_seconds) AS avg_accept_time, STDEV(delay_seconds) AS std_accept_time "
                f"FROM ride_acceptance_delay WHERE accepted_time >= '{hour_start_str}' AND accepted_time < '{hour_end_str}'",
                connection
            )
            app.logger.info(f"hourly_acceptance: {hourly_acceptance.to_dict()}")
            
            hourly_completion = pd.read_sql(
                f"SELECT AVG(duration_seconds) AS avg_ride_duration FROM ride_completion_delay "
                f"WHERE pickup_time >= '{hour_start_str}' AND pickup_time < '{hour_end_str}'",
                connection
            )
            app.logger.info(f"hourly_completion: {hourly_completion.to_dict()}")
            
            # Fetch overall stats
            overall_requests = pd.read_sql("SELECT COUNT(*) AS request_count FROM ride_requests", connection)
            overall_acceptance = pd.read_sql(
                "SELECT AVG(delay_seconds) AS avg_accept_time, STDEV(delay_seconds) AS std_accept_time "
                "FROM ride_acceptance_delay", connection
            )
            overall_completion = pd.read_sql(
                "SELECT AVG(duration_seconds) AS avg_ride_duration FROM ride_completion_delay", connection
            )
            app.logger.info(f"overall_requests: {overall_requests.to_dict()}")
            app.logger.info(f"overall_acceptance: {overall_acceptance.to_dict()}")
            app.logger.info(f"overall_completion: {overall_completion.to_dict()}")
            
            # Use hourly stats if available, otherwise fall back to overall stats
            request_count = hourly_requests['request_count'].iloc[0] if not hourly_requests.empty else (
                overall_requests['request_count'].mean() if not overall_requests.empty else 0
            )
            avg_accept_time = hourly_acceptance['avg_accept_time'].iloc[0] if not hourly_acceptance.empty else (
                overall_acceptance['avg_accept_time'].iloc[0] if not overall_acceptance.empty else 0
            )
            std_accept_time = hourly_acceptance['std_accept_time'].iloc[0] if not hourly_acceptance.empty else (
                overall_acceptance['std_accept_time'].iloc[0] if not overall_acceptance.empty else 0.001
            )
            avg_ride_duration = hourly_completion['avg_ride_duration'].iloc[0] if not hourly_completion.empty else (
                overall_completion['avg_ride_duration'].iloc[0] if not overall_completion.empty else 0
            )
            
            # Compute overall stats for z-score calculation
            overall_request_mean = overall_requests['request_count'].mean() if not overall_requests.empty else 0
            overall_request_std = overall_requests['request_count'].std() if not overall_requests.empty else 0.001
            overall_request_std = max(overall_request_std, 0.001) if not pd.isna(overall_request_std) else 0.001
            overall_accept_mean = overall_acceptance['avg_accept_time'].iloc[0] if not overall_acceptance.empty else 0
            overall_accept_std = overall_acceptance['std_accept_time'].iloc[0] if not overall_acceptance.empty else 0.001
            
            # Log z-score components
            app.logger.info(f"z_request_count components: request_count={request_count}, "
                           f"overall_request_mean={overall_request_mean}, overall_request_std={overall_request_std}")
            app.logger.info(f"z_accept_time components: current_accept_time={request.get_json()['current_accept_time']}, "
                           f"avg_accept_time={avg_accept_time}, std_accept_time={std_accept_time}")
            
            # Create mock hourly stats DataFrame with native Python types
            mock_hourly_stats = pd.DataFrame([{
                'request_count': int(request_count),  # Convert to int
                'avg_accept_time': float(avg_accept_time),  # Convert to float
                'std_accept_time': float(max(std_accept_time, 0.001)),  # Convert to float
                'avg_ride_duration': float(avg_ride_duration)  # Convert to float
            }])
            
            app.logger.info(f"Returning stats: {mock_hourly_stats.to_dict()}")
            return {
                "hourly": mock_hourly_stats,
                "overall_request_mean": float(overall_request_mean),
                "overall_request_std": float(overall_request_std),
                "overall_accept_mean": float(overall_accept_mean),
                "overall_accept_std": float(overall_accept_std)
            }
            
    except Exception as e:
        app.logger.error(f"Database query failed: {e}\nQuery details: hour_start={hour_start_str}, hour_end={hour_end_str}")
        return None

# --- Test Database Connection Endpoint ---
@app.route('/test_db', methods=['GET'])
def test_db():
    try:
        with engine.connect() as connection:
            result = connection.execute("SELECT 1 AS test").fetchone()
        app.logger.info("Database connection test successful.")
        return jsonify({"status": "success", "result": int(result[0])})
    except Exception as e:
        app.logger.error(f"Database connection test failed: {e}")
        return jsonify({"error": str(e)}), 500

# --- API Endpoint ---
@app.route('/predict_price', methods=['POST'])
def predict_price():
    if model is None:
        app.logger.error("Model not loaded.")
        return jsonify({"error": "Model not loaded. Server is not ready."}), 503
    
    if engine is None:
        app.logger.error("Database engine not initialized.")
        return jsonify({"error": "Database connection not initialized."}), 503

    try:
        data = request.get_json()
        required_fields = ['distance_km', 'current_accept_time', 'matched_at']
        if not all(field in data for field in required_fields):
            app.logger.error(f"Missing fields: {data}")
            return jsonify({"error": "Missing required fields in request body. Required: 'distance_km', 'current_accept_time', 'matched_at'."}), 400

        # Convert matched_at to datetime object
        matched_at = datetime.fromisoformat(data['matched_at'].replace('Z', '+00:00'))
        app.logger.info(f"Processing request: {data}")

        # Fetch hourly and overall stats
        stats = fetch_hourly_stats(matched_at)
        if stats is None:
            app.logger.error("Failed to fetch stats from database.")
            return jsonify({"error": "Failed to fetch required statistics from the database."}), 500

        hourly_stats = stats['hourly']
        
        # Compute z-scores
        z_request_count = (hourly_stats['request_count'].iloc[0] - stats['overall_request_mean']) / stats['overall_request_std']
        z_accept_time = (data['current_accept_time'] - hourly_stats['avg_accept_time'].iloc[0]) / hourly_stats['std_accept_time'].iloc[0]
        
        # Ensure z-scores are finite and converted to float
        z_request_count = float(z_request_count) if not pd.isna(z_request_count) else 0.0
        z_accept_time = float(z_accept_time) if not pd.isna(z_accept_time) else 0.0
        
        # Log z-scores
        app.logger.info(f"Computed z-scores: z_request_count={z_request_count}, z_accept_time={z_accept_time}")
        
        # Create a DataFrame for prediction with native Python types
        features = pd.DataFrame([{
            'distance_km': float(data['distance_km']),
            'current_accept_time': float(data['current_accept_time']),
            'request_count': int(hourly_stats['request_count'].iloc[0]),
            'avg_accept_time': float(hourly_stats['avg_accept_time'].iloc[0]),
            'avg_ride_duration': float(hourly_stats['avg_ride_duration'].iloc[0]),
            'z_request_count': z_request_count,
            'z_accept_time': z_accept_time
        }])
        
        # Log features
        app.logger.info(f"Features for prediction: {features.to_dict()}")
        
        # Predict price and convert to float
        predicted_price = float(model.predict(features)[0])
        
        # Prepare response with converted types
        response = {
            "predicted_price": round(predicted_price, 2),
            "features_used": convert_numpy_types(features.to_dict('records')[0])
        }
        
        app.logger.info(f"Predicted price: {predicted_price:.2f}")
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- Entry point ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)