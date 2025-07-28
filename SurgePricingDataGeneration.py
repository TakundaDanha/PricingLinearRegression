import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine,text
from dotenv import load_dotenv
import urllib

# Load environment variables
load_dotenv()

# SQL Server connection config
DB_HOST = os.getenv('SQLSERVER_HOST')
DB_PORT = os.getenv('SQLSERVER_PORT', '1433')
DB_NAME = os.getenv('SQLSERVER_DB')
DB_USER = os.getenv('SQLSERVER_USER')
DB_PASSWORD = os.getenv('SQLSERVER_PASSWORD')

# ODBC connection string
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

def simulate_surge_data_with_rides_week(base_fare=50.0, alpha=1.0, rides_per_hour_multiplier=25):
    np.random.seed(42)

    # Base date = last week's Monday
    base_date = datetime.now().date() - timedelta(days=7 + datetime.now().weekday())
    base_datetime = datetime.combine(base_date, datetime.min.time())

    ride_requests = []
    ride_acceptance_delays = []
    ride_completion_delays = []
    ride_matches = []

    for day in range(7):  # 7 days
        for hour in range(24):  # 24 hours per day
            # Simulate demand & supply per hour (same logic as before)
            demand = (
                10 + 30 * np.exp(-((hour - 8) ** 2) / 10) +
                40 * np.exp(-((hour - 17) ** 2) / 12) +
                np.random.randint(0, 5)
            )
            supply = (
                15 + 20 * np.exp(-((hour - 9) ** 2) / 15) +
                30 * np.exp(-((hour - 18) ** 2) / 10) +
                np.random.randint(0, 4)
            )
            supply = max(supply, 1)
            ratio = demand / supply
            surge_multiplier = np.clip(1 + alpha * (ratio - 1), 1, 3)
            

            num_rides = max(1, int(demand * rides_per_hour_multiplier / 100))

            for _ in range(num_rides):
                ride_id = str(uuid.uuid4())
                rider_id = f"rider_{random.randint(0, 999)}"
                driver_id = f"driver_{random.randint(0, 499)}"

                minute = random.randint(0, 59)
                second = random.randint(0, 59)

                request_time = base_datetime + timedelta(days=day, hours=hour, minutes=minute, seconds=second)
                accept_delay = random.randint(int(30 * surge_multiplier), int(300 * surge_multiplier))
                accepted_time = request_time + timedelta(seconds=accept_delay)

                ride_duration = random.randint(300, 2700)
                pickup_time = accepted_time
                completed_at = pickup_time + timedelta(seconds=ride_duration)

                pickup_lat = -26.1076 + random.random() * 0.5
                pickup_lng = 28.0567 + random.random() * 0.5
                dropoff_lat = -26.1150 + random.random() * 0.5
                dropoff_lng = 28.0620 + random.random() * 0.5
                base_ride_price = 12.0 + random.random() * 10
                surge_price = round(base_ride_price * surge_multiplier, 2)

                ride_requests.append({
                    'ride_id': ride_id,
                    'rider_id': rider_id,
                    'pickup_lat': pickup_lat,
                    'pickup_lng': pickup_lng,
                    'dropoff_lat': dropoff_lat,
                    'dropoff_lng': dropoff_lng,
                    'request_time': request_time
                })

                ride_acceptance_delays.append({
                    'ride_id': ride_id,
                    'rider_id': rider_id,
                    'driver_id': driver_id,
                    'request_time': request_time,
                    'accepted_time': accepted_time,
                    'delay_seconds': accept_delay
                })

                ride_completion_delays.append({
                    'ride_id': ride_id,
                    'driver_id': driver_id,
                    'pickup_time': pickup_time,
                    'completed_at': completed_at,
                    'duration_seconds': ride_duration
                })

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
                    'price': surge_price
                })

    # Convert to DataFrames
    df_ride_requests = pd.DataFrame(ride_requests)
    df_ride_acceptance_delays = pd.DataFrame(ride_acceptance_delays)
    df_ride_completion_delays = pd.DataFrame(ride_completion_delays)
    df_ride_matches = pd.DataFrame(ride_matches)

    return df_ride_requests, df_ride_acceptance_delays, df_ride_completion_delays, df_ride_matches


def clear_existing_data():
    tables_to_clear = ['ride_requests', 'ride_acceptance_delay', 'ride_completion_delay', 'ride_matches']
    
    try:
        with engine.begin() as connection:  # begin() handles commit/rollback automatically
            for table in tables_to_clear:
                sql = text(f"DELETE FROM dbo.{table}")
                connection.execute(sql)
                print(f"Cleared all data from {table}")
        print("All tables cleared successfully.")
    except Exception as e:
        print(f"Error clearing tables: {e}")
        raise

def insert_surge_ride_data(df_requests, df_acceptance, df_completion, df_matches):
   
    try:
        with engine.connect() as connection:
            df_requests.to_sql('ride_requests', connection, schema='dbo', if_exists='append', index=False)
            df_acceptance.to_sql('ride_acceptance_delay', connection, schema='dbo', if_exists='append', index=False)
            df_completion.to_sql('ride_completion_delay', connection, schema='dbo', if_exists='append', index=False)
            df_matches.to_sql('ride_matches', connection, schema='dbo', if_exists='append', index=False)
        
        total_rides = len(df_requests)
        print(f"Successfully inserted {total_rides} surge-based rides into all tables.")
        
        # Print surge statistics
        avg_price = df_matches['price'].mean()
        max_price = df_matches['price'].max()
        min_price = df_matches['price'].min()
        print(f"Price stats - Avg: R{avg_price:.2f}, Min: R{min_price:.2f}, Max: R{max_price:.2f}")
        
    except Exception as e:
        print(f"Error inserting surge pricing ride data: {e}")
        raise

if __name__ == "__main__":
    # Clear existing data first
    print("Clearing existing data from all tables...")
    clear_existing_data()
    
    # Generate surge-based ride data
    print("Generating new surge-based ride data...")
    df_requests, df_acceptance, df_completion, df_matches = simulate_surge_data_with_rides_week(
    base_fare=50.0, 
    alpha=1.0, 
    rides_per_hour_multiplier=25
)

    
    # Insert into the same tables as the original script
    print("Inserting new data...")
    insert_surge_ride_data(df_requests, df_acceptance, df_completion, df_matches)