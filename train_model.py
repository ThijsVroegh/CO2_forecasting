import os
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import argparse

import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

import read_ned
from config import MODEL_DIR, HISTORICAL_DIR, TRAINING_DAYS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to the DataFrame.
    
    Adds:
    1. Temporal features:
        - Hour of day (0-23)
        - Day of week (0=Monday, 6=Sunday)
        - Month (1-12)
        - Is weekend (1 for Sat/Sun, 0 otherwise)
        
    2. Lagged features:
        - Short-term: 1, 2, 3 hours
        - Daily: 24, 48 hours (1-2 days)
        - Weekly: 168 hours (7 days)
        
    3. Rolling means:
        - Daily (24h)
        - Weekly (168h)
    """
    df = df.copy()
    
    # Add temporal features
    df['hour'] = pd.to_datetime(df.index).hour
    df['dayofweek'] = pd.to_datetime(df.index).dayofweek  # Monday=0, Sunday=6
    df['month'] = pd.to_datetime(df.index).month    
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Add lag features for emission factor
    lags = [1, 2, 3, 24, 48, 168]  # hours
    for lag in lags:
        df[f'emissionfactor_lag_{lag}h'] = df['emissionfactor'].shift(lag)
    
    # Add rolling means for different windows
    windows = [24, 168]  # 1 day, 1 week
    for window in windows:
        df[f'emissionfactor_rolling_mean_{window}h'] = (
            df['emissionfactor'].rolling(window=window, min_periods=1).mean()
        )
    
    # Drop rows with NaN values from the lag creation
    df = df.dropna()
    
    return df


def gluonify(df: pd.DataFrame) -> TimeSeriesDataFrame:
    """Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame format."""
    df = df.copy()
    df = df.reset_index()
    df["item_id"] = 0
    
    # Ensure the timestamp column is named 'timestamp'
    if 'validfrom (UTC)' in df.columns:
        df = df.rename(columns={'validfrom (UTC)': 'timestamp'})
    elif 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})
        
    return TimeSeriesDataFrame.from_data_frame(
        df=df,
        timestamp_column="timestamp",
        id_column="item_id"
    )


def load_training_data() -> pd.DataFrame:
    """Load and optionally filter historical data for training."""
    ned_data = read_ned.read_all(HISTORICAL_DIR)
    ned_data.index = pd.to_datetime(ned_data.index)
    
    if TRAINING_DAYS is not None:
        # Calculate cutoff date
        cutoff_date = ned_data.index[-1] - pd.Timedelta(days=TRAINING_DAYS)
        ned_data = ned_data[ned_data.index >= cutoff_date]
        print(f"Using data from {ned_data.index[0]} to {ned_data.index[-1]}")
        print(f"Total training period: {TRAINING_DAYS} days")
    else:
        print(f"Using all available data: {ned_data.index[0]} to {ned_data.index[-1]}")
    
    return ned_data


if __name__ == "__main__":
    # Load data with configured training period
    ned_data = load_training_data()

    # Add all features to the dataset
    ned_data_with_features = add_features(ned_data)
    
    # Split into train and test sets
    # train_data is all data except the last week
    train_data = ned_data_with_features[:-7*24]
    
    # test_data is the last week of available data
    test_data = ned_data_with_features[-7*24:]

    # Convert to AutoGluon TimeSeriesDataFrame format
    gluon_train_data = gluonify(train_data)
    gluon_test_data = gluonify(test_data)

    # known_covariates are vars that will be "known" in the forecasting horizon
    predictor = TimeSeriesPredictor(
        prediction_length=7*24,
        freq="1h",
        target="emissionfactor",
        # Include both renewable volumes and temporal features as known covariates
        known_covariates_names=[
            # Renewable energy volumes (from NED forecasts)
            "volume_sun",
            "volume_land-wind", 
            "volume_sea-wind",
            # Temporal features (can be calculated for any date)
            "hour",
            "dayofweek",
            "month",
            "is_weekend"
        ],
        # Lag features and rolling means will be used as static features
        path=str(MODEL_DIR)
    ).fit(
        gluon_train_data,
        time_limit=200,
        presets="best_quality",
        excluded_model_types=["Chronos", "DeepAR", "TiDE"]
    )
    
    leaderboard = predictor.leaderboard(gluon_train_data, silent=True)
    print("\nModel Leaderboard:")
    print(leaderboard)
    
    # Print feature names to verify lag features were added
    print("\nFeatures used in training:")
    print(gluon_train_data.columns.tolist())
    
    # Calculate and plot feature importance
    compute_importance = True  # Set to True to compute feature importance
    if compute_importance:
        print("\nComputing feature importance (this may take a while)...")
        print("Press Ctrl+C to skip feature importance calculation")
        try:
            fimportance = TimeSeriesPredictor.feature_importance(
                predictor,
                subsample_size=1  # Since we only have one time series
            )
            
            fimportance = fimportance.sort_values('importance')

            plt.figure(figsize=(12,5))
            plt.barh(fimportance.index, fimportance['importance'])
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            
            # Save feature importance plot
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            plt.savefig(MODEL_DIR / f'feature_importance_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance data
            fimportance.to_csv(MODEL_DIR / f'feature_importance_{timestamp}.csv')
            print("Feature importance analysis completed and saved!")
            
        except KeyboardInterrupt:
            print("\nFeature importance calculation skipped by user")
        except Exception as e:
            print(f"Warning: Could not compute feature importance: {str(e)}")

# # Load and merge temperature data
#     try:
#         temp_df = pd.read_csv('knmi_data/processed_temperatures.csv')
#         temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
#         temp_df = temp_df.set_index('datetime')
        
#         # Merge temperature data with main dataframe
#         df = df.join(temp_df, how='left')
        
#         # Add temperature lag features
#         temp_lags = [1, 2, 3, 24]  # hours
#         for lag in temp_lags:
#             df[f'temperature_lag_{lag}h'] = df['temperature_celsius'].shift(lag)
        
#         # Add temperature rolling means
#         temp_windows = [24, 168]  # 1 day, 1 week
#         for window in temp_windows:
#             df[f'temperature_rolling_mean_{window}h'] = (
#                 df['temperature_celsius'].rolling(window=window, min_periods=1).mean()
#             )
#     except Exception as e:
#         print(f"Warning: Could not load temperature data: {str(e)}")