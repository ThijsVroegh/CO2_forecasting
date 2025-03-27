from typing import Dict, List, Union, Set
import datetime
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from workalendar.europe import Netherlands, NetherlandsWithSchoolHolidays

from meteo_utils import combine_meteo_data
import read_ned
from config import MODEL_DIR, HISTORICAL_DIR, TRAINING_DAYS


def get_dutch_holidays(year: int) -> Dict[date, str]:
    """Return Dutch national holidays for a given year.
    
    Args:
        year: The year to get holidays for
        
    Returns:
        Dictionary mapping holiday dates to holiday names
    """
    cal = Netherlands()
    holidays = {}
    
    # Get holidays
    for holiday_date, holiday_name in cal.holidays(year):
        holidays[holiday_date] = holiday_name    
    return holidays


def get_dutch_school_vacations(year: int) -> Dict[str, List[Dict[str, date]]]:
    """Return Dutch school vacation periods for a given year.
    
    Args:
        year: The year to get vacations for
        
    Returns:
        Dictionary with vacation periods by region
    """
    # Create instances for each region
    regions = {
        "north": NetherlandsWithSchoolHolidays(region="north"),
        "middle": NetherlandsWithSchoolHolidays(region="middle"),
        "south": NetherlandsWithSchoolHolidays(region="south")
    }
    
    # Initialize dictionary
    vacations = {}
    
    # Process each region
    for region_name, cal in regions.items():
        # Get all variable days (includes school holidays)
        all_days = cal.get_variable_days(year)
        
        # Filter for school holidays only and group by type
        for day_date, day_name in all_days:
            # Extract vacation type from name (e.g., "Fall holiday" -> "fall")
            if "holiday" in day_name.lower():
                vacation_type = day_name.lower().replace(" holiday", "")
                
                if vacation_type not in vacations:
                    vacations[vacation_type] = {}
                
                if region_name not in vacations[vacation_type]:
                    # Initialize with this date as both start and end
                    vacations[vacation_type][region_name] = {
                        'start': day_date,
                        'end': day_date
                    }
                else:
                    # Update end date if this date is later
                    if day_date < vacations[vacation_type][region_name]['start']:
                        vacations[vacation_type][region_name]['start'] = day_date
                    if day_date > vacations[vacation_type][region_name]['end']:
                        vacations[vacation_type][region_name]['end'] = day_date
    return vacations

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Dutch holiday features to the dataframe.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with DateTime index
        
    Returns:
        pd.DataFrame: DataFrame with added holiday features
    """
    df = df.copy()
    datetime_index = pd.to_datetime(df.index)
    
    # Initialize holiday columns
    df['is_holiday'] = 0
    
    # Store the name of the holiday
    df['holiday_name'] = ''
    
    # Get unique years in the data
    years = datetime_index.year.unique()
    
    # Create holiday indicators
    for year in years:
        holidays = get_dutch_holidays(year)
        for holiday_date, holiday_name in holidays.items():
            # Mark holidays
            holiday_mask = (datetime_index.date == holiday_date)
            df.loc[holiday_mask, 'is_holiday'] = 1
            df.loc[holiday_mask, 'holiday_name'] = holiday_name
        
    df['is_working_day'] = 0
    cal = Netherlands()
    
    for date_idx in pd.date_range(df.index.min().date(), df.index.max().date()):
        # Only process each date once to improve performance
        mask = (datetime_index.date == date_idx.date())
        if mask.any():
            is_working = cal.is_working_day(date_idx.date())
            df.loc[mask, 'is_working_day'] = int(is_working)    
    return df


def add_vacation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Dutch school vacation features to the dataframe.
    For regions, we use a combined approach where a day is marked as vacation
    if it is a vacation in any region.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with DateTime index
        
    Returns:
        pd.DataFrame: DataFrame with added vacation features
    """
    df = df.copy()
    datetime_index = pd.to_datetime(df.index)
    
    # Initialize vacation columns
    df['is_school_vacation'] = 0
    df['is_summer_vacation'] = 0
    df['vacation_type'] = 'none'
    
    # Get unique years in the data
    years = datetime_index.year.unique()
    
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
                
                # If a day is already marked as vacation, keep the existing vacation_type
                # unless the new one is 'summer' which takes precedence
                new_vacation_days = (
                    (vacation_mask) & 
                    ((df['vacation_type'] == 'none') | 
                     (vacation_type == 'summer' and df['vacation_type'] != 'summer'))
                )
                
                df.loc[vacation_mask, 'is_school_vacation'] = 1
                df.loc[new_vacation_days, 'vacation_type'] = vacation_type
                
                # Mark summer vacation specifically
                if vacation_type == 'summer':
                    df.loc[vacation_mask, 'is_summer_vacation'] = 1    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic temporal features to the dataframe.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with DateTime index
        
    Returns:
        pd.DataFrame: DataFrame with added temporal features
    """
    df = df.copy()
    datetime_index = pd.to_datetime(df.index)
        
    df['hour'] = datetime_index.hour
    df['dayofweek'] = datetime_index.dayofweek
    df['month'] = datetime_index.month    
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonal cyclical features using sine and cosine transformations.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with DateTime index and basic temporal features
        
    Returns:
        pd.DataFrame: DataFrame with added seasonal features
    """
    df = df.copy()
    datetime_index = pd.to_datetime(df.index)
    
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
    
    return df


def add_emission_factor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add emission factor derived features if 'emissionfactor' column is present.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with possible 'emissionfactor' column
        
    Returns:
        pd.DataFrame: DataFrame with added emission factor features
    """
    df = df.copy()
    
    if 'emissionfactor' in df.columns:        
        df['emissionfactor_diff'] = df['emissionfactor'].diff()
        
        # lag features for emission factor based on ACF plot
        lags = [1, 2, 3, 24, 48, 168]
        for lag in lags:
            df[f'emissionfactor_lag_{lag}h'] = df['emissionfactor'].shift(lag)
        
        # rolling means for emission factor
        windows = [24, 168]
        for window in windows:
            df[f'emissionfactor_rolling_mean_{window}h'] = (
                df['emissionfactor'].rolling(window=window, min_periods=1).mean()
            )
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add electricity and gas price features including energy crisis indicators.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with DateTime index
        
    Returns:
        pd.DataFrame: DataFrame with added price features
    """
    df = df.copy()
    
    try:
        script_dir = MODEL_DIR.parent
        price_features_file = script_dir / "data" / "price_features.csv"
        
        if price_features_file.exists():
            print(f"Loading price features from {price_features_file}")
            price_df = pd.read_csv(price_features_file, index_col=0, parse_dates=True)
            
            # Verify data coverage
            print("\nPrice data coverage check:")
            print(f"Input data period: {df.index.min()} to {df.index.max()}")
            print(f"Price data period: {price_df.index.min()} to {price_df.index.max()}")
            
            # Merge price features with data
            # Only include base features that would be known for future forecast periods
            price_features_to_include = [
                'elec_price', 'gas_price', 'gas_to_elec_ratio',
                'elec_price_diff_1d','elec_price_diff_1w',
                'energy_crisis','phase_pre_crisis', 'phase_early_crisis',
                'phase_acute_crisis', 'phase_peak_crisis', 'phase_stabilization'
            ]
            
            if set(price_features_to_include).issubset(price_df.columns):                
                df = pd.merge(df, price_df[price_features_to_include], 
                             left_index=True, right_index=True, 
                             how='left', validate='1:1')  # Ensure one-to-one merge
                
                # Check for NaN values from merge
                nan_count = df['elec_price'].isna().sum()
                if nan_count > 0:
                    print(f"\nWarning: Found {nan_count} NaN values in electricity price after merge")
                    # Fill NaN values with forward fill then backward fill
                    df[price_features_to_include] = df[price_features_to_include].ffill().bfill()
            else:
                missing_cols = set(price_features_to_include) - set(price_df.columns)
                print(f"Warning: Missing required price features: {missing_cols}")                
    
    except Exception as e:
        print(f"Warning: Could not add price features: {str(e)}")
        print("Continuing without price features...")
    
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive weather features including temperature, precipitation, wind, etc.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with DateTime index
        
    Returns:
        pd.DataFrame: DataFrame with added weather features
    """
    df = df.copy()
    
    try:
        weather_df = combine_meteo_data()
        
        # Verify data coverage
        print("\nData coverage check:")
        print(f"ned_data period: {df.index.min()} to {df.index.max()}")
        print(f"weather_data period: {weather_df.index.min()} to {weather_df.index.max()}")
        
        # Initialize dict to collect all new features
        feature_dict = {}
        
        # Basic temperature features
        if 'temperature' in weather_df.columns:
            # Get temperature data
            temp_df = pd.merge(df, weather_df[['temperature']], 
                             left_index=True, right_index=True, 
                             how='left',
                             validate='1:1')  # Ensure one-to-one merge
            
            # Check for unexpected NaN values
            nan_count = temp_df['temperature'].isna().sum()
            if nan_count > 0:
                print(f"\nWarning: Found {nan_count} unexpected NaN values in temperature after merge")
            
            # Add temperature to feature dict
            feature_dict['temperature'] = temp_df['temperature']
            
            # Temperature changes
            feature_dict['temperature_change_1h'] = temp_df['temperature'].diff()
            feature_dict['temperature_change_24h'] = temp_df['temperature'].diff(24)
            
            # Temperature lags
            temp_lags = [1, 8, 9, 10, 20, 21, 22, 23, 24]
            for lag in temp_lags:
                feature_dict[f'temperature_lag_{lag}h'] = temp_df['temperature'].shift(lag)
            
            # Temperature rolling means
            temp_windows = [24, 168]  # 1 day, 1 week
            for window in temp_windows:
                feature_dict[f'temperature_rolling_mean_{window}h'] = (
                    temp_df['temperature'].rolling(window=window, min_periods=1).mean()
                )
        else:
            print("Warning: temperature not found in weather data")
            feature_dict['temperature'] = 0
        
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
                # Get the feature data from weather_df
                feature_data = pd.merge(df, weather_df[[source_col]],
                                      left_index=True, right_index=True,
                                      how='left',
                                      validate='1:1')[source_col]
                
                if source_col == 'is_day':
                    # Boolean feature
                    feature_dict[target_col] = feature_data
                else:
                    # Numeric features
                    feature_dict[target_col] = feature_data
                    
                    # Add changes except for boolean features
                    feature_dict[f'{target_col}_change_1h'] = feature_data.diff()
                    feature_dict[f'{target_col}_change_24h'] = feature_data.diff(24)
                    
                    # Add rolling means
                    for window in temp_windows:
                        feature_dict[f'{target_col}_rolling_mean_{window}h'] = (
                            feature_data.rolling(window=window, min_periods=1).mean()
                        )
            else:
                print(f"Warning: {source_col} not found in weather data")
        
        # Combine the original DataFrame with all new features at once
        all_features_df = pd.DataFrame(feature_dict, index=df.index)
        result_df = pd.concat([df, all_features_df], axis=1)
        
        return result_df
            
    except Exception as e:
        print(f"Warning: Could not add weather features: {str(e)}")
        print("Continuing without weather features...")

    return df


def add_features(
    df: pd.DataFrame, 
    feature_sets: Union[str, Set[str]] = 'all'
) -> pd.DataFrame:
    """
    Combine multiple feature sets based on the specified options.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with DateTime index
        feature_sets (Union[str, Set[str]]): Specification of feature sets to include.
            - 'all': Include all feature sets
            - Set of strings: Only include specified feature sets
              Options: {'temporal', 'holiday', 'vacation', 'seasonal', 
                       'emission', 'price', 'weather'}
    
    Returns:
        pd.DataFrame: DataFrame with selected features added
    
    Example:
        # Include all features
        df_all_features = add_features(df, 'all')
        
        # Include only temporal and seasonal features
        df_selected = add_features(df, {'temporal', 'seasonal'})
    """
    df = df.copy()
    
    # Define all available feature sets
    all_feature_sets = {'temporal', 'holiday', 'vacation', 'seasonal', 'emission', 'price', 'weather'}
    
    # Determine which feature sets to include
    if feature_sets == 'all':
        selected_feature_sets = all_feature_sets
    else:
        # Validate feature sets
        if not isinstance(feature_sets, set):
            raise ValueError("feature_sets must be 'all' or a set of feature types")
        
        invalid_sets = feature_sets - all_feature_sets
        if invalid_sets:
            raise ValueError(f"Invalid feature sets: {invalid_sets}. "
                            f"Valid options are: {all_feature_sets}")
        
        selected_feature_sets = feature_sets
    
    # Apply selected feature transformations
    if 'temporal' in selected_feature_sets:
        df = add_temporal_features(df)
    
    if 'holiday' in selected_feature_sets:
        df = add_holiday_features(df)
    
    if 'vacation' in selected_feature_sets:
        df = add_vacation_features(df)
    
    if 'seasonal' in selected_feature_sets and 'temporal' in selected_feature_sets:
        # Seasonal features depend on temporal features
        df = add_seasonal_features(df)
    elif 'seasonal' in selected_feature_sets:
        # If temporal features weren't selected but seasonal were,
        # we need to add temporal first
        temp_df = add_temporal_features(df)
        df = add_seasonal_features(temp_df)
        # Keep only the seasonal features, not the temporal ones
        temporal_cols = ['hour', 'dayofweek', 'month', 'is_weekend']
        df = df.drop(columns=[col for col in temporal_cols if col in df.columns])
    
    if 'emission' in selected_feature_sets:
        df = add_emission_factor_features(df)
    
    if 'price' in selected_feature_sets:
        df = add_price_features(df)
    
    if 'weather' in selected_feature_sets:
        df = add_weather_features(df)
    
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
    
    ned_data = load_training_data()   
    print("Columns ned_data:", ned_data.columns.tolist())
    
    # Split into train and test sets
    train_data = ned_data[:-7*24]  # All data except last week available in the dataset
    test_data = ned_data[-7*24:]   # Last week reserved for testing

    # Add features
    
    # use all available features
    #train_data_with_features = add_features(train_data, 'all')
    
    # exclude price features
    train_data_with_features = add_features(train_data, {'temporal', 'holiday', 'vacation', 'seasonal', 'emission', 'weather'})
    train_data_with_features.head()
    print("Columns train_data_with_features:", train_data_with_features.columns.tolist())     
    
    # Convert to AutoGluon format
    gluon_train_data = gluonify(train_data_with_features)
    
    # Print feature names to verify lag features are present
    print("\nFeatures used in training:")
    print(gluon_train_data.columns.tolist())
    
    predictor = TimeSeriesPredictor(
        prediction_length=7*24,
        freq="1h",
        target="emissionfactor",
        path=str(MODEL_DIR),
        # known_covariates are variables which are known in the forecasting horizon
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
            "is_holiday", "is_working_day",
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
            # Price features
            #"elec_price", "gas_price", "gas_to_elec_ratio", 'elec_price_diff_1d','elec_price_diff_1w',
            # Energy crisis indicators
            # "energy_crisis",
            # "phase_pre_crisis", "phase_early_crisis", 
            # "phase_acute_crisis", "phase_peak_crisis", "phase_stabilization",
            ]
    ).fit(
        train_data      = gluon_train_data,
        presets         = "best_quality",
        time_limit      = 1000,
        num_val_windows = 3,
    )    

    leaderboard = predictor.leaderboard(gluon_train_data, silent=True)
    print("\nModel Leaderboard:")
    print(leaderboard)
            
    # Feature importance ----
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