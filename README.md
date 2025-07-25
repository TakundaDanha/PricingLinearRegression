## PricingLinearRegression

A machine learning project to derive coefficients for a dynamic pricing algorithm using regression models, leveraging surge-based ride data to reflect real-world demand patterns.

## Overview

This project analyzes ride-sharing data to model the relationship between ride features (e.g., distance, demand, acceptance time) and prices. It improves upon uniform data generation with surge-based data, enabling a dynamic pricing system that mirrors traffic congestion patterns.

## Features

- **Data Loading and Preprocessing**:
  - Loads data from SQL Server tables (`ride_requests`, `ride_acceptance_delay`, `ride_completion_delay`, `ride_matches`).
  - Computes features: Euclidean distance (`distance_km`), request hour/day, hourly demand stats (e.g., request count, z-scores).
  - Cleans data (e.g., datetime conversion, comma-to-decimal fixes).
- **Data Generation**:
  - Generates surge-based ride data with demand peaks at 8 AM and 5 PM, replacing uniform distributions.
- **Model Training**:
  - Trains Linear, Ridge, Lasso, and ElasticNet regression models (with cross-validation).
  - Targets ride price (ZAR) for dynamic pricing.
- **Evaluation**:
  - Evaluates models using Mean Squared Error (MSE) and R² Score.
  - Provides coefficient interpretations for pricing algorithms.
- **Output**:
  - Exports pricing coefficients for SQL stored procedures (e.g., `baseRate`, `coeffRequests`).
  - Saves the best model and provides sample predictions.

## Models

### Linear Regression
Fits a linear model to predict prices, minimizing squared errors.

### Ridge Regression
Uses L2 regularization to reduce overfitting with correlated features.

### Lasso Regression
Applies L1 regularization for feature selection, potentially zeroing out coefficients.

### ElasticNet Regression
Combines L1 and L2 penalties for balanced regularization and feature selection.

## Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn sqlalchemy pyodbc python-dotenv
```

Configure SQL Server connection in a `.env` file:

```plaintext
SQLSERVER_HOST=your_host
SQLSERVER_PORT=1433
SQLSERVER_DB=your_database
SQLSERVER_USER=your_user
SQLSERVER_PASSWORD=your_password
```

## Usage

1. **Generate Surge-Based Data**:
   ```bash
   python SurgePricingDataGeneration.py
   ```

2. **Train and Evaluate Models**:
   ```bash
   python PriceTargetLinearRegression.py
   ```

## Scripts

- **`SurgePricingDataGeneration.py`**:
  - Replaces `generate_ride_data.py` (uniform distribution).
  - Simulates surge pricing with demand peaks at 8 AM and 5 PM, based on demand-to-supply ratios.
  - Outputs data to SQL tables with price statistics.

- **`PriceTargetLinearRegression.py`**:
  - Enhances `linearRegression.py` by targeting price (ZAR) instead of duration.
  - Loads data from ride tables, computes features (e.g., distance, demand stats), and trains regression models.
  - Outputs pricing coefficients and sample predictions.

## Output

- **Console**:
  - Model metrics (R², MSE).
  - Coefficient interpretations (e.g., price per km).
  - SQL `DECLARE` statements for pricing stored procedures.
  - Sample prediction (actual vs. predicted price).
- **Database**:
  - Populated ride tables with surge-based data.

## Improvements

- **Realistic Data**: Uses surge-based data with demand peaks, replacing uniform distributions.
- **Price Focus**: Targets price for dynamic pricing, not duration.
- **Enhanced Features**: Includes distance, demand stats, and z-scores.
- **Robust Database**: Adds connection timeouts and pooling.

## Future Enhancements

- Add factors like weather or events to surge pricing.
- Extend data generation for multi-day/week trends.
- Use real-world geospatial data for locations.
- Implement advanced pricing models.
```
