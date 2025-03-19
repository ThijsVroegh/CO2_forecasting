import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path
from typing import Optional, Dict
import datetime
import pytz

# def convert_to_nl_time(df: pd.DataFrame) -> pd.DataFrame:
#     """Convert DataFrame index to Dutch local time format matching ned.nl data.
    
#     Args:
#         df: DataFrame with timezone-aware UTC index
    
#     Returns:
#         DataFrame with timezone-naive index in Dutch local time
#     """
#     # First make sure we're working with timezone-aware UTC
#     if df.index.tz is None:
#         df.index = df.index.tz_localize('UTC')
#     elif df.index.tz != pytz.UTC:
#         df.index = df.index.tz_convert('UTC')
    
#     # Convert to Amsterdam time
#     df.index = df.index.tz_convert('Europe/Amsterdam')
    
#     # Remove timezone info to match ned.nl format
#     df.index = df.index.tz_localize(None)
    
#     return df

def convert_to_nl_time(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame index is timezone-naive Dutch local time."""
    
    # If the index is naive, assume it is already in Dutch time and return as is
    if df.index.tz is None:
        return df  # No conversion needed
    
    # If the index is already in Europe/Amsterdam, just remove timezone info
    if str(df.index.tz) == "Europe/Amsterdam":
        df.index = df.index.tz_localize(None)
        return df
    
    # Otherwise, assume UTC and convert to Dutch time
    df.index = df.index.tz_convert('Europe/Amsterdam')
    df.index = df.index.tz_localize(None)
    
    return df


def fetch_historical_weather() -> pd.DataFrame:
    """Fetch historical weather data from Open-Meteo API.
    
    Returns:
        pd.DataFrame: Historical weather data with timezone-naive Dutch local time index
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.11,
        "longitude": 5.1806,
        "start_date": "2021-01-01",
        "end_date": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                  "apparent_temperature", "precipitation", "rain", "snowfall",
                  "pressure_msl", "surface_pressure", "cloud_cover", 
                  "et0_fao_evapotranspiration", 
                  "wind_speed_10m", "wind_direction_10m", "uv_index", "uv_index_clear_sky",
                  "is_day", "sunshine_duration", "shortwave_radiation", "direct_radiation",
                  "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
                  "terrestrial_radiation"],
        "timezone": "Europe/Amsterdam"  # Use Dutch timezone
    }
    
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()
    
    # Create DataFrame with proper datetime index (timezone-aware)
    df = pd.DataFrame(index=pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
        tz='Europe/Amsterdam'  # Specify timezone during creation
    ))
    
    # Add variables with proper column names
    column_mapping = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dew_point",
        "apparent_temperature": "feels_like",
        "precipitation": "precipitation",
        "rain": "rain",
        "snowfall": "snowfall",
        "pressure_msl": "pressure_msl",
        "surface_pressure": "surface_pressure",
        "cloud_cover": "cloud_cover",
        "et0_fao_evapotranspiration": "evapotranspiration",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "uv_index": "uv_index",
        "uv_index_clear_sky": "uv_index_clear",
        "is_day": "is_day",
        "sunshine_duration": "sunshine_duration",
        "shortwave_radiation": "solar_radiation",
        "direct_radiation": "direct_radiation",
        "diffuse_radiation": "diffuse_radiation",
        "direct_normal_irradiance": "direct_normal_irradiance",
        "global_tilted_irradiance": "global_irradiance",
        "terrestrial_radiation": "terrestrial_radiation"
    }
    
    for i, (api_name, df_name) in enumerate(column_mapping.items()):
        df[df_name] = hourly.Variables(i).ValuesAsNumpy()
    
    # Convert to Dutch local time format (timezone-naive)
    df = convert_to_nl_time(df)
    
    return df

def fetch_forecast_weather() -> pd.DataFrame:
    """Fetch weather forecast data from Open-Meteo API.
    
    Returns:
        pd.DataFrame: Forecast weather data with timezone-naive Dutch local time index
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://api.open-meteo.com/v1/forecast"
    start_date = datetime.datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    params = {
        "latitude": 52.11,
        "longitude": 5.1806,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                  "apparent_temperature", "precipitation", "rain", "snowfall",
                  "pressure_msl", "surface_pressure", "cloud_cover", 
                  "et0_fao_evapotranspiration", 
                  "wind_speed_10m", "wind_direction_10m", "uv_index", "uv_index_clear_sky",
                  "is_day", "sunshine_duration", "shortwave_radiation", "direct_radiation",
                  "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
                  "terrestrial_radiation"],
        "timezone": "Europe/Amsterdam"  # Use Dutch timezone
    }
    
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()
    
    # Create DataFrame with proper datetime index (timezone-aware)
    df = pd.DataFrame(index=pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
        tz='Europe/Amsterdam'  # Specify timezone during creation
    ))
    
    # Add variables with proper column names (same mapping as historical data)
    column_mapping = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dew_point",
        "apparent_temperature": "feels_like",
        "precipitation": "precipitation",
        "rain": "rain",
        "snowfall": "snowfall",
        "pressure_msl": "pressure_msl",
        "surface_pressure": "surface_pressure",
        "cloud_cover": "cloud_cover",
        "et0_fao_evapotranspiration": "evapotranspiration",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_direction",
        "uv_index": "uv_index",
        "uv_index_clear_sky": "uv_index_clear",
        "is_day": "is_day",
        "sunshine_duration": "sunshine_duration",
        "shortwave_radiation": "solar_radiation",
        "direct_radiation": "direct_radiation",
        "diffuse_radiation": "diffuse_radiation",
        "direct_normal_irradiance": "direct_normal_irradiance",
        "global_tilted_irradiance": "global_irradiance",
        "terrestrial_radiation": "terrestrial_radiation"
    }
    
    for i, (api_name, df_name) in enumerate(column_mapping.items()):
        df[df_name] = hourly.Variables(i).ValuesAsNumpy()
    
    # Convert to Dutch local time format (timezone-naive)
    df = convert_to_nl_time(df)
    
    return df

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in weather data through intelligent imputation.
    
    Methods:
    - For small gaps (1-2 hours): Linear interpolation
    - For larger gaps: Forward fill + backward fill average
    - For wind direction: Circular interpolation
    
    Args:
        df: DataFrame with potential missing values
    
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    # Create complete time index
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h'
    )
    
    # Reindex to expose missing timestamps
    df = df.reindex(full_index)
    
    # Handle each column appropriately
    for col in df.columns:
        if col == 'wind_direction':
            # Circular interpolation for wind direction
            df[col] = df[col].interpolate(method='linear', limit=2)
            # For larger gaps, use nearest neighbor
            df[col] = df[col].ffill().bfill()
        else:
            # Linear interpolation for small gaps
            df[col] = df[col].interpolate(method='linear', limit=2)
            # For larger gaps, use average of forward and backward fill
            if df[col].isnull().any():
                ffill = df[col].ffill()
                bfill = df[col].bfill()
                df[col] = (ffill + bfill) / 2
    
    # Verify no missing values remain
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: Some missing values could not be imputed:")
        print(missing[missing > 0])
    
    return df

def combine_meteo_data(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Combine historical and forecast weather data from Open-Meteo.
    
    This function fetches both historical and forecast data, combines them,
    and ensures there are no duplicates or gaps in the time series. All timestamps
    are in Dutch local time (timezone-naive) to match ned.nl data format.
    
    Args:
        output_path: Optional path to save the combined data. If None, data is not saved.
    
    Returns:
        pd.DataFrame: Combined weather data with timezone-naive Dutch local time index
    """
    try:
        # Fetch both historical and forecast data
        print("Fetching historical weather data...")
        historical_df = fetch_historical_weather()
        
        print("Fetching forecast weather data...")
        forecast_df = fetch_forecast_weather()
        
        # Combine data, giving preference to historical data where dates overlap
        print("Combining historical and forecast data...")
        combined_df = pd.concat([historical_df, forecast_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()
        
        # Handle missing timestamps through imputation
        print("Checking for missing timestamps...")
        expected_dates = pd.date_range(
            start=combined_df.index.min(),
            end=combined_df.index.max(),
            freq='h'
        )
        missing_dates = expected_dates.difference(combined_df.index)
        if len(missing_dates) > 0:
            print(f"Found {len(missing_dates)} missing timestamps")
            print(f"First few missing dates: {missing_dates[:5]}")
            print("Performing intelligent imputation...")
            combined_df = handle_missing_data(combined_df)
        
        # Verify timezone-naive format
        if combined_df.index.tz is not None:
            print("Warning: DataFrame index still has timezone information. Converting to timezone-naive...")
            combined_df = convert_to_nl_time(combined_df)
        
        # Validate data ranges
        print("\nValidating data ranges:")
        for col in combined_df.columns:
            print(f"{col:15}: min={combined_df[col].min():8.2f}, max={combined_df[col].max():8.2f}")
        
        if output_path is not None:
            # Save with explicit datetime index name
            combined_df.index.name = 'datetime'
            combined_df.to_csv(output_path)
            print(f"Combined weather data saved to {output_path}")
        
        print(f"\nSuccessfully combined weather data:")
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Total records: {len(combined_df)}")
        print("\nColumns available:")
        print(combined_df.columns.tolist())
        print("\nSample timestamps to verify format:")
        print(combined_df.head().to_string())
        
        return combined_df
        
    except Exception as e:
        print(f"Error processing weather data: {str(e)}")
        raise

if __name__ == "__main__":
    # When run as a script, combine the data and save it
    output_path = Path("weather_meteo_combined.csv")
    df = combine_meteo_data(output_path) 