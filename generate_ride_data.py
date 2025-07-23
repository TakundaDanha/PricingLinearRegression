import os
import uuid
import random
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib

# Load environment variables
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


connection_uri = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_str)}"
engine = create_engine(connection_uri)

def generate_ride_data(num_rides=600):
    # Base date: last week's Monday
    base_date = (datetime.now().date() - timedelta(days=7 + datetime.now().weekday())).strftime('%Y-%m-%d')
    base_date = datetime.strptime(base_date, '%Y-%m-%d')

    ride_requests = []
    ride_acceptance_delays = []
    ride_completion_delays = []
    ride_matches = []

    for _ in range(num_rides):
        # Generate IDs
        ride_id = str(uuid.uuid4())
        rider_id = f'rider_{random.randint(0, 999)}'
        driver_id = f'driver_{random.randint(0, 499)}'

        # Generate random time components
        day_offset = random.randint(0, 6)  # Any day from last week
        hour = random.randint(0, 23)  # 5am to 11pm
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        # Calculate request time
        request_time = base_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        # Calculate delays and times
        accept_delay = random.randint(0, 300)  # 0 to 5 minutes
        accepted_time = request_time + timedelta(seconds=accept_delay)
        ride_duration = random.randint(300, 2700)  # 5 to 45 minutes
        pickup_time = accepted_time
        completed_at = pickup_time + timedelta(seconds=ride_duration)

        # Generate coordinates (Lagos-like)
        pickup_lat = 6.5244 + random.random() * 0.1
        pickup_lng = 3.3792 + random.random() * 0.1
        dropoff_lat = 6.4654 + random.random() * 0.1
        dropoff_lng = 3.4064 + random.random() * 0.1

        # Price calculation
        price = round(12.0 + random.random() * 10, 2)

        # Collect data for ride_requests
        ride_requests.append({
            'ride_id': ride_id,
            'rider_id': rider_id,
            'pickup_lat': pickup_lat,
            'pickup_lng': pickup_lng,
            'dropoff_lat': dropoff_lat,
            'dropoff_lng': dropoff_lng,
            'request_time': request_time
        })

        # Collect data for ride_acceptance_delay
        ride_acceptance_delays.append({
            'ride_id': ride_id,
            'rider_id': rider_id,
            'driver_id': driver_id,
            'request_time': request_time,
            'accepted_time': accepted_time,
            'delay_seconds': accept_delay
        })

        # Collect data for ride_completion_delay
        ride_completion_delays.append({
            'ride_id': ride_id,
            'driver_id': driver_id,
            'pickup_time': pickup_time,
            'completed_at': completed_at,
            'duration_seconds': ride_duration
        })

        # Collect data for ride_matches
        ride_matches.append({
            'ride_id': ride_id,
            'rider_id': rider_id,
            'driver_id': driver_id,
            'pickup_lat': pickup_lat,
            'pickup_lng': pickup_lng,
            'dropoff_lat': dropoff_lat,
            'dropoff_lng': dropoff_lng,
            'matched_at': accepted_time,
            'status': 'completed',
            'driver_response_at': accepted_time,
            'started_at': pickup_time,
            'arrived_at': pickup_time,
            'completed_at': completed_at,
            'price': price
        })

    # Convert to DataFrames
    df_ride_requests = pd.DataFrame(ride_requests)
    df_ride_acceptance_delays = pd.DataFrame(ride_acceptance_delays)
    df_ride_completion_delays = pd.DataFrame(ride_completion_delays)
    df_ride_matches = pd.DataFrame(ride_matches)

    # Write to database
    try:
        with engine.connect() as connection:
            df_ride_requests.to_sql('ride_requests', connection, schema='dbo', if_exists='append', index=False)
            df_ride_acceptance_delays.to_sql('ride_acceptance_delay', connection, schema='dbo', if_exists='append', index=False)
            df_ride_completion_delays.to_sql('ride_completion_delay', connection, schema='dbo', if_exists='append', index=False)
            df_ride_matches.to_sql('ride_matches', connection, schema='dbo', if_exists='append', index=False)
        print(f"Inserted {num_rides} records into all tables successfully.")
    except Exception as e:
        print(f"Error inserting data into database: {e}")
        raise

if __name__ == "__main__":
    generate_ride_data()