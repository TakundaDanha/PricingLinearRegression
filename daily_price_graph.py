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

def plot_price_distribution(selected_date_str):
    selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()

    query_prices = f"""
        SELECT matched_at, price
        FROM dbo.ride_matches
        WHERE CAST(matched_at AS DATE) = '{selected_date}' and  Status = 'completed'
    """

    df_prices = pd.read_sql(query_prices, engine)

    if df_prices.empty:
        print(f"No price data available for {selected_date}")
        return

    # Extract hour from timestamps and ensure price is numeric
    df_prices['hour'] = pd.to_datetime(df_prices['matched_at']).dt.hour
    df_prices['price'] = pd.to_numeric(df_prices['price'], errors='coerce')

    # Calculate average price by hour
    hourly_prices = df_prices.groupby('hour')['price'].mean().reindex(range(24), fill_value=0)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_prices, label='Average Ride Price', marker='o', color='green')
    plt.title(f"Average Ride Price Distribution on {selected_date}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Price ($)")
    plt.xticks(range(24))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this to the date you'd like to visualize
    date_to_plot = input("Enter date to visualize (YYYY-MM-DD): ").strip()
    plot_price_distribution(date_to_plot)