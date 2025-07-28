import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import urllib
from math import radians, sin, cos, sqrt, atan2

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

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in kilometers using the Haversine formula."""
    R = 6371.0  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def simulate_surge_data_with_rides_week(base_fare_per_km=6.88, alpha=1.0, rides_per_hour_multiplier=50):
    np.random.seed(42)

    # Base date = last week's Monday
    base_date = datetime.now().date() - timedelta(days=7 + datetime.now().weekday())
    base_datetime = datetime.combine(base_date, datetime.min.time())

    ride_requests = []
    ride_acceptance_delays = []
    ride_completion_delays = []
    ride_matches = []

    # Coordinates for Sandton City and Bryanpark
    base_pickup_lat, base_pickup_lng = -26.1076, 28.0567
    base_dropoff_lat, base_dropoff_lng = -26.1150, 28.0620

    for day in range(7):  # 7 days
        for hour in range(24):  # 24 hours per day
            # Enhanced demand model with sharper peaks at 8 AM and 5 PM
            demand = (
                20 +  # Base demand
                60 * np.exp(-((hour - 8) ** 2) / 8) +  # Morning peak
                80 * np.exp(-((hour - 17) ** 2) / 6) +  # Evening peak
                np.random.randint(0, 10)  # Random variation
            )
            # Supply lags demand during peaks
            supply = (
                15 +  # Base supply
                40 * np.exp(-((hour - 9) ** 2) / 10) +  # Morning supply peak
                50 * np.exp(-((hour - 18) ** 2) / 8) +  # Evening supply peak
                np.random.randint(0, 8)  # Random variation
            )
            supply = max(supply, 1)
            ratio = demand / supply
            surge_multiplier = np.clip(1 + alpha * (ratio - 1), 1, 3)

            # Number of ride requests based on demand
            num_rides = max(1, int(demand * rides_per_hour_multiplier / 100))
            # Completion rate: 90–95% during peak hours (7–9 AM, 4–6 PM), 95–100% otherwise
            completion_rate = 0.95 if (7 <= hour <= 9 or 16 <= hour <= 18) else 0.98
            num_completed = int(num_rides * (completion_rate + np.random.uniform(-0.05, 0.05)))
            num_completed = max(1, min(num_rides, num_completed))

            for i in range(num_rides):
                ride_id = str(uuid.uuid4())
                rider_id = f"rider_{random.randint(0, 999)}"
                driver_id = f"driver_{random.randint(0, 499)}" if i < num_completed else "driver_unmatched"

                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                request_time = base_datetime + timedelta(days=day, hours=hour, minutes=minute, seconds=second)

                # Use Sandton City and Bryanpark coordinates with variation
                pickup_lat = base_pickup_lat + random.random() * 0.5 - 0.25
                pickup_lng = base_pickup_lng + random.random() * 0.5 - 0.25
                dropoff_lat = base_dropoff_lat + random.random() * 0.5 - 0.25
                dropoff_lng = base_dropoff_lng + random.random() * 0.5 - 0.25

                # Calculate distance and ensure ~10.5 km
                distance_km = haversine_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
                if distance_km < 8.0 or distance_km > 13.0:
                    scale_factor = 10.5 / distance_km
                    pickup_lat = base_pickup_lat + (pickup_lat - base_pickup_lat) * scale_factor
                    pickup_lng = base_pickup_lng + (pickup_lng - base_pickup_lng) * scale_factor
                    dropoff_lat = base_dropoff_lat + (dropoff_lat - base_dropoff_lat) * scale_factor
                    dropoff_lng = base_dropoff_lng + (dropoff_lng - base_dropoff_lng) * scale_factor
                    distance_km = 10.5

                # Calculate base price
                base_ride_price = base_fare_per_km * distance_km + random.uniform(-3, 3)  # ±R3 variation
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

                if i < num_completed:
                    accept_delay = random.randint(int(30 * surge_multiplier), int(300 * surge_multiplier))
                    accepted_time = request_time + timedelta(seconds=accept_delay)
                    ride_duration = random.randint(300, 2700)
                    pickup_time = accepted_time
                    completed_at = pickup_time + timedelta(seconds=ride_duration)

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
                # else:
                #     ride_matches.append({
                #         'ride_id': ride_id,
                #         'rider_id': rider_id,
                #         'driver_id': driver_id,  # Use placeholder driver_id
                #         'pickup_lat': pickup_lat,
                #         'pickup_lng': pickup_lng,
                #         'dropoff_lat': dropoff_lat,
                #         'dropoff_lng': dropoff_lng,
                #         'matched_at': None,
                #         'status': 'pending',
                #         'driver_response_at': None,
                #         'started_at': None,
                #         'arrived_at': None,
                #         'completed_at': None,
                #         'price': 0.0
                #     })

    # Convert to DataFrames
    df_ride_requests = pd.DataFrame(ride_requests)
    df_ride_acceptance_delays = pd.DataFrame(ride_acceptance_delays)
    df_ride_completion_delays = pd.DataFrame(ride_completion_delays)
    df_ride_matches = pd.DataFrame(ride_matches)

    return df_ride_requests, df_ride_acceptance_delays, df_ride_completion_delays, df_ride_matches

def clear_existing_data():
    tables_to_clear = ['ride_requests', 'ride_acceptance_delay', 'ride_completion_delay', 'ride_matches']
    
    try:
        with engine.begin() as connection:
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
        completed_rides = len(df_matches[df_matches['status'] == 'completed'])
        print(f"Successfully inserted {total_rides} rides ({completed_rides} completed, {total_rides - completed_rides} unmatched).")
        
        # Print surge statistics for completed rides
        completed_matches = df_matches[df_matches['status'] == 'completed']
        avg_price = completed_matches['price'].mean()
        max_price = completed_matches['price'].max()
        min_price = completed_matches['price'].min()
        print(f"Price stats (completed rides) - Avg: R{avg_price:.2f}, Min: R{min_price:.2f}, Max: R{max_price:.2f}")
        
    except Exception as e:
        print(f"Error inserting surge pricing ride data: {e}")
        raise

if __name__ == "__main__":
    # Clear existing data
    print("Clearing existing data from all tables...")
    #clear_existing_data()
    
    # Generate surge-based ride data
    print("Generating new surge-based ride data...")
    df_requests, df_acceptance, df_completion, df_matches = simulate_surge_data_with_rides_week(
        base_fare_per_km=6.88, 
        alpha=1.0, 
        rides_per_hour_multiplier=50
    )
    
    # Insert into tables
    print("Inserting new data...")
    insert_surge_ride_data(df_requests, df_acceptance, df_completion, df_matches)