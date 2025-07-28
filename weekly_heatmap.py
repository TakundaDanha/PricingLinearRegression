import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
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


def fetch_weekly_data(start_date):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = start + timedelta(days=6)

    query_requests = f"""
        SELECT request_time
        FROM dbo.ride_requests
        WHERE CAST(request_time AS DATE) BETWEEN '{start}' AND '{end}'
    """

    query_completions = f"""
        SELECT completed_at
        FROM dbo.ride_completion_delay
        WHERE CAST(completed_at AS DATE) BETWEEN '{start}' AND '{end}'
    """

    df_requests = pd.read_sql(query_requests, engine)
    df_completions = pd.read_sql(query_completions, engine)

    df_requests['day'] = pd.to_datetime(df_requests['request_time']).dt.date
    df_requests['hour'] = pd.to_datetime(df_requests['request_time']).dt.hour

    df_completions['day'] = pd.to_datetime(df_completions['completed_at']).dt.date
    df_completions['hour'] = pd.to_datetime(df_completions['completed_at']).dt.hour

    return df_requests, df_completions


def plot_weekly_heatmap(df, value_label):
    # Create pivot table
    pivot = df.groupby(['day', 'hour']).size().unstack(fill_value=0)
    pivot = pivot.reindex(sorted(pivot.index), axis=0)

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5, linecolor='white', annot=True, fmt='d')
    plt.title(f"Hourly Ride {value_label} Heatmap (per Day)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Date")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    week_start = input("Enter Monday's date of the week to visualize (YYYY-MM-DD): ").strip()
    df_requests, df_completions = fetch_weekly_data(week_start)

    if df_requests.empty and df_completions.empty:
        print("No data found for this week.")
    else:
        if not df_requests.empty:
            print("Showing ride requests heatmap...")
            plot_weekly_heatmap(df_requests, "Requests")

        if not df_completions.empty:
            print("Showing ride completions heatmap...")
            plot_weekly_heatmap(df_completions, "Completions")
