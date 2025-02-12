#import os
#from pathlib import Path
import matplotlib.pyplot as plt
import datetime
#import argparse
from datetime import date
from typing import Dict, List

import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from dateutil.easter import easter

import read_ned
from config import MODEL_DIR, HISTORICAL_DIR, TRAINING_DAYS


def get_dutch_holidays(year: int) -> Dict[date, str]:
    """Return Dutch national holidays for a given year."""
    easter_date = easter(year)
    
    holidays = {
        # Fixed dates
        date(year, 1, 1): "Nieuwjaarsdag",  
        date(year, 4, 27): "Koningsdag",    
        date(year, 5, 5): "Bevrijdingsdag", 
        date(year, 12, 25): "Eerste Kerstdag", 
        date(year, 12, 26): "Tweede Kerstdag", 
                
        easter_date: "Eerste Paasdag",
        easter_date + pd.Timedelta(days=1): "Tweede Paasdag", 
        easter_date + pd.Timedelta(days=39): "Hemelvaartsdag", 
        easter_date + pd.Timedelta(days=49): "Eerste Pinksterdag",
        easter_date + pd.Timedelta(days=50): "Tweede Pinksterdag",
    }
    return holidays


def get_dutch_school_vacations(year: int) -> Dict[str, List[Dict[str, date]]]:
    """Return Dutch school vacation periods for a given year.
    
    Note: These are approximate dates, actual dates may vary by region and year.
    Returns periods for North, Middle, and South regions.
    """
    vacations = {
        # Christmas vacation (2 weeks, all regions same dates)
        "christmas": {
            "all": {
                "start": date(year, 12, 25),
                "end": date(year + 1, 1, 9)
            }
        },
        # Spring vacation (1 week, different per region)
        "spring": {
            "north": {
                "start": date(year, 2, 19),
                "end": date(year, 2, 27)
            },
            "middle": {
                "start": date(year, 2, 26),
                "end": date(year, 3, 6)
            },
            "south": {
                "start": date(year, 2, 19),
                "end": date(year, 2, 27)
            }
        },
        # May vacation (1-2 weeks)
        "may": {
            "all": {
                "start": date(year, 4, 29),
                "end": date(year, 5, 7)
            }
        },
        # Summer vacation (6 weeks, different per region)
        "summer": {
            "north": {
                "start": date(year, 7, 22),
                "end": date(year, 9, 3)
            },
            "middle": {
                "start": date(year, 7, 8),
                "end": date(year, 8, 20)
            },
            "south": {
                "start": date(year, 7, 15),
                "end": date(year, 8, 27)
            }
        },
        # Autumn vacation (1 week)
        "autumn": {
            "all": {
                "start": date(year, 10, 14),
                "end": date(year, 10, 22)
            }
        }
    }
    return vacations


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """   
    This function adds a variety of features to the DataFrame, including:
    - Temporal features: hour, day of the week, month, and whether the day is a weekend.
    - Holiday features: indicators for Dutch national holidays and adjacent days.
    - Vacation features: indicators for Dutch school vacations, including summer vacations.
    - Seasonal features: sine and cosine transformations of day of the year, week of the year, 
      hour of the day, day of the week, month, and quarter.
    - Emission factor features: differences, lags, and rolling means if the 'emissionfactor' 
      column is present.
    - Weather features: temperature, solar radiation, wind speed, wind direction, cloud cover, 
      and relative humidity, if available from external weather data.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a DateTime index and optional 'emissionfactor' column.

    Returns:
    pd.DataFrame: A new DataFrame with the original data and additional features.
    
    Notes:
    - The function assumes the input DataFrame has a DateTime index.
    - Weather data is expected to be in 'data/knmi_data/processed_weather_data.csv'. This data is from
    https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens, from de Bilt weather station.    
    - The function modifies a copy of the input DataFrame to avoid altering the original data.
    """
        
    df = df.copy()
    
    # Add temporal features
    datetime_index = pd.to_datetime(df.index)
    df['hour'] = datetime_index.hour
    df['dayofweek'] = datetime_index.dayofweek
    df['month'] = datetime_index.month    
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Add holiday features
    df['is_holiday'] = 0
    df['is_holiday_adjacent'] = 0  # Day before/after holiday
    
    # Add vacation features
    df['is_school_vacation'] = 0
    df['is_summer_vacation'] = 0
    df['vacation_type'] = 'none'
    
    # Get unique years in the data
    years = datetime_index.year.unique()
    
    # Create holiday indicators
    for year in years:
        holidays = get_dutch_holidays(year)
        for holiday_date, holiday_name in holidays.items():
            # Mark holidays
            holiday_mask = (datetime_index.date == holiday_date)
            df.loc[holiday_mask, 'is_holiday'] = 1
            
            # Mark adjacent days (day before and after)
            adjacent_dates = [
                holiday_date - pd.Timedelta(days=1),
                holiday_date + pd.Timedelta(days=1)
            ]
            for adj_date in adjacent_dates:
                adj_mask = (datetime_index.date == adj_date)
                df.loc[adj_mask, 'is_holiday_adjacent'] = 1
    
    # Create vacation indicators
    for year in years:
        vacations = get_dutch_school_vacations(year)
        
        for vacation_type, regions in vacations.items():
            for region, period in regions.items():
                start_date = period['start']
                end_date = period['end']
                
                # Mark vacation days
                vacation_mask = (
                    (datetime_index.date >= start_date) & 
                    (datetime_index.date <= end_date)
                )
                df.loc[vacation_mask, 'is_school_vacation'] = 1
                df.loc[vacation_mask, 'vacation_type'] = vacation_type
                
                # Mark summer vacation specifically
                if vacation_type == 'summer':
                    df.loc[vacation_mask, 'is_summer_vacation'] = 1
    
    # Add seasonal features
    # Day of year (1-366)
    df['day_of_year'] = datetime_index.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Week of year (1-53)
    df['week_of_year'] = datetime_index.isocalendar().week
    df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52.1775)
    df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52.1775)
    
    # Hour of day (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (0-6)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Month (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Quarter (1-4)
    df['quarter'] = datetime_index.quarter
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Add emission factor difference and lags    
    if 'emissionfactor' in df.columns:  
        # ensures no emission factor features are added to forecast data,
        # only for historical data
        df['emissionfactor_diff'] = df['emissionfactor'].diff()
        
        # Add lag features for emission factor
        lags = [1, 2, 3, 24, 48, 168]  # hours
        for lag in lags:
            df[f'emissionfactor_lag_{lag}h'] = df['emissionfactor'].shift(lag)
        
        # Add rolling means for emission factor
        windows = [24, 168]  # 1 day, 1 week
        for window in windows:
            df[f'emissionfactor_rolling_mean_{window}h'] = (
                df['emissionfactor'].rolling(window=window, min_periods=1).mean()
            )
    
    # Add temperature features if available
    try:
        # Read weather data with all KNMI variables
        weather_df = pd.read_csv('data/knmi_data/processed_weather_data.csv')
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
        weather_df = weather_df.set_index('datetime')
        
        # Add temperature and its derived features
        df['temperature'] = weather_df['temperature'].reindex(df.index)
        df['temperature_change_1h'] = df['temperature'].diff()
        df['temperature_change_24h'] = df['temperature'].diff(24)
        
        # Add temperature lag features
        important_lags = [1, 8, 9, 10, 20, 21, 22, 23, 24]
        for lag in important_lags:
            df[f'temperature_lag_{lag}h'] = df['temperature'].shift(lag)
        
        # Add temperature rolling means
        df['temperature_rolling_mean_24h'] = df['temperature'].rolling(window=24, min_periods=1).mean()
        df['temperature_rolling_mean_168h'] = df['temperature'].rolling(window=168, min_periods=1).mean()
        
        # Add other KNMI features
        df['solar_radiation'] = weather_df['solar_radiation'].reindex(df.index)
        df['wind_speed'] = weather_df['wind_speed'].reindex(df.index)
        df['wind_direction'] = weather_df['wind_direction'].reindex(df.index)
        df['cloud_cover'] = weather_df['cloud_cover'].reindex(df.index)
        df['relative_humidity'] = weather_df['relative_humidity'].reindex(df.index)
        
    except FileNotFoundError:
        print("Warning: Weather data not found, temperature features will not be added")
    
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

    # Split into train and test sets
    train_data = ned_data[:-7*24]  # All data except last week
    test_data = ned_data[-7*24:]   # Last week for testing

    # Add features - training data gets all features
    train_data_with_features = add_features(train_data)
    
    # Test data only gets features available in forecast horizon
    test_data_with_features = add_features(test_data)

    # Convert to AutoGluon format
    gluon_train_data = gluonify(train_data_with_features)
    gluon_test_data = gluonify(test_data_with_features)

    # known_covariates are vars that will be "known" in the forecasting horizon
    predictor = TimeSeriesPredictor(
        prediction_length=7*24,
        freq="1h",
        target="emissionfactor",
        known_covariates_names=[
            # Renewable energy volumes
            "volume_sun", "volume_land-wind", "volume_sea-wind",
            # Basic temporal features
            "hour", "dayofweek", "month", "is_weekend",
            # Cyclical features - Day of year
            "day_of_year_sin", "day_of_year_cos",
            # Cyclical features - Week of year
            "week_of_year_sin", "week_of_year_cos",
            # Cyclical features - Hour
            "hour_sin", "hour_cos",
            # Cyclical features - Day of week
            "dayofweek_sin", "dayofweek_cos",
            # Cyclical features - Month
            "month_sin", "month_cos",
            # Cyclical features - Quarter
            "quarter_sin", "quarter_cos",
            # Temperature features
            "temperature", "temperature_change_1h", "temperature_change_24h",
            # Temperature lag features
            'temperature_lag_1h', 'temperature_lag_8h', 'temperature_lag_9h', 'temperature_lag_10h', 'temperature_lag_20h', 'temperature_lag_21h', 'temperature_lag_22h', 'temperature_lag_23h', 'temperature_lag_24h',
            # Temperature rolling means
            'temperature_rolling_mean_24h', 'temperature_rolling_mean_168h',
            # Weather features
            "solar_radiation", "wind_speed", "wind_direction", "cloud_cover", "relative_humidity",
            # Holiday features
            "is_holiday", "is_holiday_adjacent",
            # Vacation features
            "is_school_vacation", "is_summer_vacation", "vacation_type",                        
        ],
        path=str(MODEL_DIR)
    ).fit(
        train_data=gluon_train_data,        
        presets="best_quality",
        verbosity=4,
        time_limit=1000,
        num_val_windows=3, # Used to select best model during training
        #excluded_model_types=["Chronos", "DeepAR", "TiDE"]
    )
    

    leaderboard = predictor.leaderboard(gluon_train_data, silent=True)
    print("\nModel Leaderboard:")
    print(leaderboard)
    
    # Print feature names to verify lag features were added
    print("\nFeatures used in training:")
    print(gluon_train_data.columns.tolist())
    
    # Calculate and plot feature importance
    compute_importance = False  # Set to True to compute feature importance
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