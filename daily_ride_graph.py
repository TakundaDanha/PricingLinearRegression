import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine
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


def plot_ride_distribution(selected_date_str):
    selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()

    query_requests = f"""
        SELECT request_time
        FROM dbo.ride_requests
        WHERE CAST(request_time AS DATE) = '{selected_date}'
    """

    query_completions = f"""
        SELECT completed_at
        FROM dbo.ride_completion_delay
        WHERE CAST(completed_at AS DATE) = '{selected_date}'
    """

    df_requests = pd.read_sql(query_requests, engine)
    df_completions = pd.read_sql(query_completions, engine)

    if df_requests.empty and df_completions.empty:
        print(f"No data available for {selected_date}")
        return

    # Extract hour from timestamps
    df_requests['hour'] = pd.to_datetime(df_requests['request_time']).dt.hour
    df_completions['hour'] = pd.to_datetime(df_completions['completed_at']).dt.hour

    # Count rides by hour
    hourly_requests = df_requests.groupby('hour').size().reindex(range(24), fill_value=0)
    hourly_completions = df_completions.groupby('hour').size().reindex(range(24), fill_value=0)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_requests, label='Ride Requests', marker='o')
    plt.plot(hourly_completions, label='Ride Completions', marker='s')
    plt.title(f"Ride Distribution on {selected_date}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Rides")
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change this to the date you'd like to visualize
    date_to_plot = input("Enter date to visualize (YYYY-MM-DD): ").strip()
    plot_ride_distribution(date_to_plot)
