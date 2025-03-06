from typing import Dict, List
import datetime
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.common import space
from dateutil.easter import easter
from meteo_utils import combine_meteo_data
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
    This function adds a variety of features to the dataframe, including:
    - Temporal features: hour, day of the week, month, and whether the day is a weekend.
    - Holiday features: indicators for Dutch national holidays and adjacent days.
    - Vacation features: indicators for Dutch school vacations, including summer vacations.
    - Seasonal features: sine and cosine transformations of day of the year, week of the year, 
      hour of the day, day of the week, month, and quarter.
    - Emission factor features: differences, lags, and rolling means if the 'emissionfactor' 
      column is present.
    - Weather features: comprehensive set of weather variables from Open-Meteo API including
      temperature, humidity, precipitation, wind, cloud cover, and solar radiation.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a DateTime index and optional 'emissionfactor' column.

    Returns:
    pd.DataFrame: A new DataFrame with the original data and additional features.
    
    Notes:
    - The function assumes the input DataFrame has a DateTime index.
    - Weather data is expected to be available from Open-Meteo API through meteo_utils.py
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
    
    # Add weather features
    try:        
        weather_df = combine_meteo_data()
        
        # Verify data coverage
        print("\nData coverage check:")
        print(f"ned_data period: {df.index.min()} to {df.index.max()}")
        print(f"weather_data period: {weather_df.index.min()} to {weather_df.index.max()}")
        
        # Basic temperature features
        if 'temperature' in weather_df.columns:
            # Use merge with validate to ensure data quality
            df = pd.merge(df, weather_df[['temperature']], 
                         left_index=True, right_index=True, 
                         how='left',
                         validate='1:1')  # Ensure one-to-one merge
            
            # Check for unexpected NaN values
            nan_count = df['temperature'].isna().sum()
            if nan_count > 0:
                print(f"\nWarning: Found {nan_count} unexpected NaN values in temperature after merge")
        else:
            print("Warning: temperature not found in weather data")
            df['temperature'] = 0
        
        # Temperature changes
        df['temperature_change_1h'] = df['temperature'].diff()
        df['temperature_change_24h'] = df['temperature'].diff(24)
        
        # Temperature lags
        temp_lags = [1, 8, 9, 10, 20, 21, 22, 23, 24]
        for lag in temp_lags:
            df[f'temperature_lag_{lag}h'] = df['temperature'].shift(lag)
        
        # Temperature rolling means
        temp_windows = [24, 168]  # 1 day, 1 week
        for window in temp_windows:
            df[f'temperature_rolling_mean_{window}h'] = (
                df['temperature'].rolling(window=window, min_periods=1).mean()
            )
        
        # Additional weather features from open-meteo
        weather_features = {
            'humidity': 'humidity',
            'dew_point': 'dew_point',
            'feels_like': 'feels_like',
            'precipitation': 'precipitation',
            'rain': 'rain',
            'snowfall': 'snowfall',
            'pressure_msl': 'pressure_msl',
            'surface_pressure': 'surface_pressure',
            'cloud_cover': 'cloud_cover',
            'evapotranspiration': 'evapotranspiration',
            'wind_speed': 'wind_speed',
            'wind_direction': 'wind_direction',
            'uv_index': 'uv_index',
            'uv_index_clear': 'uv_index_clear',
            'is_day': 'is_day',
            'sunshine_duration': 'sunshine_duration',
            'solar_radiation': 'solar_radiation',
            'direct_radiation': 'direct_radiation',
            'diffuse_radiation': 'diffuse_radiation',
            'direct_normal_irradiance': 'direct_normal_irradiance',
            'global_irradiance': 'global_irradiance',
            'terrestrial_radiation': 'terrestrial_radiation'
        }
        
        for source_col, target_col in weather_features.items():
            if source_col in weather_df.columns:                
                temp_df = pd.merge(df, weather_df[[source_col]],
                                 left_index=True, right_index=True,
                                 how='left',
                                 validate='1:1')
                
                if source_col == 'is_day':
                    # Boolean feature
                    df[target_col] = temp_df[source_col]
                else:
                    # Numeric features
                    df[target_col] = temp_df[source_col]
                    
                    # Add changes except for boolean features
                    df[f'{target_col}_change_1h'] = df[target_col].diff()
                    df[f'{target_col}_change_24h'] = df[target_col].diff(24)
                    
                    # Add rolling means
                    for window in temp_windows:
                        df[f'{target_col}_rolling_mean_{window}h'] = (
                            df[target_col].rolling(window=window, min_periods=1).mean()
                        )
            else:
                print(f"Warning: {source_col} not found in weather data")
            
    except Exception as e:
        print(f"Warning: Could not add weather features: {str(e)}")
        print("Continuing without weather features...")

    return df

def gluonify(df: pd.DataFrame) -> TimeSeriesDataFrame:
    """Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame format."""
    df = df.copy()
    
    # Reset index and ensure it becomes a column named 'timestamp'
    df = df.reset_index()
    df["item_id"] = 0
    
    # Ensure the timestamp column is named 'timestamp'
    if 'validfrom (UTC)' in df.columns:
        df = df.rename(columns={'validfrom (UTC)': 'timestamp'})
    elif 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    elif 'index' in df.columns:
        df = df.rename(columns={'index': 'timestamp'})
    
    # If still no timestamp column, check if the index was reset properly
    if 'timestamp' not in df.columns:
        print("Warning: No timestamp column found. Current columns:", df.columns.tolist())
        # Try to identify the datetime column
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            print(f"Found datetime column: {datetime_cols[0]}")
            df = df.rename(columns={datetime_cols[0]: 'timestamp'})
        else:
            raise ValueError("No suitable timestamp column found in the DataFrame")
    
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
    train_data = ned_data[:-7*24]  # All data except last week available in the dataset 
    test_data = ned_data[-7*24:]   # Last week reserved for testing

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
        path=str(MODEL_DIR),
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
            # Holiday features
            "is_holiday", "is_holiday_adjacent",
            # Vacation features
            "is_school_vacation", "is_summer_vacation", "vacation_type",
            # Wind features
            "wind_speed", "wind_speed_change_1h",
            "wind_speed_rolling_mean_24h",  
            "wind_speed_rolling_mean_168h",            
            "wind_direction", "wind_direction_change_1h",
            "wind_direction_rolling_mean_24h",  
            "wind_direction_rolling_mean_168h",                                    
            # Cloud
            'cloud_cover', 'cloud_cover_change_1h', 'cloud_cover_change_24h',
            "cloud_cover_rolling_mean_24h",  
            "cloud_cover_rolling_mean_168h",        
            # Solar/Radiation features
            "solar_radiation", 
            "solar_radiation_rolling_mean_24h",
            "solar_radiation_rolling_mean_168h",
            "direct_radiation",
            "direct_radiation_rolling_mean_24h",
            "direct_radiation_rolling_mean_168h",            
            # Humidity
            'humidity', 'humidity_change_1h', 'humidity_change_24h',
            "humidity_rolling_mean_24h",  
            "humidity_rolling_mean_168h",            
            # Precipitation
            'precipitation', 'precipitation_change_1h', 'precipitation_change_24h',
            "precipitation_rolling_mean_24h",  
            "precipitation_rolling_mean_168h", 
            ]
    ).fit(
        train_data=gluon_train_data,        
        presets="best_quality",
        time_limit=1000,
        num_val_windows=3,  # This will reduce the likelihood of overfitting
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
            
            # Debug: Print feature importance values
            print("Feature Importance Values:")
            print(fimportance)
            
            fimportance = fimportance.sort_values('importance')

            plt.figure(figsize=(12,15))
            plt.barh(fimportance.index, fimportance['importance'])
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
                        
            # Save feature importance plot
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            plt.savefig(MODEL_DIR / f'feature_importance_{timestamp}.png', 
                       dpi=600, bbox_inches='tight')
            
            # Show plot only after saving
            plt.show() 
            
            plt.close()
            
            # Save feature importance data
            fimportance.to_csv(MODEL_DIR / f'feature_importance_{timestamp}.csv')
            print("Feature importance analysis completed and saved!")
            
        except KeyboardInterrupt:
            print("\nFeature importance calculation skipped by user")
        except Exception as e:
            print(f"Warning: Could not compute feature importance: {str(e)}")