import os
import logging
from pathlib import Path
import requests
from datetime import datetime, timedelta
import pandas as pd
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherAPI:
    """Client for fetching weather data from OpenWeatherMap."""
    
    def __init__(self, api_key=None):
        """Initialize OpenWeatherMap API client.
        
        Args:
            api_key (str): Your OpenWeatherMap API key. If not provided, will look for OPENWEATHER_API_KEY env variable.
        """
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenWeatherMap API key is required. Either pass it to the constructor or "
                "set the OPENWEATHER_API_KEY environment variable."
            )
        
        # Create data directory if it doesn't exist
        self.data_dir = Path("data/weather_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default coordinates for De Bilt, Netherlands (KNMI main weather station; we take this
        # temperature as the reference temperature for the Netherlands)
        self.default_lat = 52.1093
        self.default_lon = 5.1810
    
    def kelvin_to_celsius(self, kelvin):
        """Convert Kelvin to Celsius."""
        return kelvin - 273.15
    
    def fetch_current_weather(self, lat=None, lon=None):
        """Fetch current weather data.
        
        Args:
            lat (float): Latitude (default: De Bilt)
            lon (float): Longitude (default: De Bilt)
        """
        lat = lat or self.default_lat
        lon = lon or self.default_lon
        
        # Build URL
        url = f"{self.base_url}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"  # Get temperature in Celsius
        }
        
        # Make request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def fetch_forecast(self, lat=None, lon=None):
        """Fetch 5-day weather forecast data.
        
        Args:
            lat (float): Latitude (default: De Bilt)
            lon (float): Longitude (default: De Bilt)
        """
        lat = lat or self.default_lat
        lon = lon or self.default_lon
        
        # Build URL
        url = f"{self.base_url}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"  # Get temperature in Celsius
        }
        
        # Make request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def process_weather_data(self, current_data, forecast_data):
        """Process weather data into a pandas DataFrame."""
        records = []
        
        # Process current weather
        current = {
            'datetime': datetime.fromtimestamp(current_data['dt']),
            'temperature': current_data['main']['temp'],
            'feels_like': current_data['main']['feels_like'],
            'humidity': current_data['main']['humidity'],
            'pressure': current_data['main']['pressure'],
            'wind_speed': current_data['wind']['speed'],
            'wind_direction': current_data['wind'].get('deg', None),
            'clouds': current_data['clouds']['all'],
            'weather_main': current_data['weather'][0]['main'],
            'weather_description': current_data['weather'][0]['description'],
            'is_forecast': False
        }
        records.append(current)
        
        # Process forecast data
        for item in forecast_data['list']:
            forecast = {
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'feels_like': item['main']['feels_like'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'wind_direction': item['wind'].get('deg', None),
                'clouds': item['clouds']['all'],
                'weather_main': item['weather'][0]['main'],
                'weather_description': item['weather'][0]['description'],
                'is_forecast': True
            }
            records.append(forecast)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        return df
    
    def fetch_weather_data(self):
        """Fetch both current weather and forecast data."""
        logger.info("Fetching weather data from OpenWeatherMap")
        
        try:
            # Fetch current weather and forecast
            current_data = self.fetch_current_weather()
            forecast_data = self.fetch_forecast()
            
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            raw_data = {
                "current": current_data,
                "forecast": forecast_data,
                "metadata": {
                    "location": "De Bilt, Netherlands",
                    "latitude": self.default_lat,
                    "longitude": self.default_lon,
                    "timestamp": timestamp
                }
            }
            
            # Save raw JSON
            raw_file = self.data_dir / f"weather_data_{timestamp}.json"
            with open(raw_file, 'w') as f:
                json.dump(raw_data, f, indent=2)
            
            # Process data into DataFrame
            df = self.process_weather_data(current_data, forecast_data)
            
            # Save processed CSV
            csv_file = self.data_dir / f"weather_data_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            # Show information about the data
            logger.info("\nData Information:")
            logger.info(f"Location: De Bilt, Netherlands ({self.default_lat}, {self.default_lon})")
            logger.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
            logger.info(f"Number of records: {len(df)}")
            logger.info("\nVariables available:")
            logger.info("- temperature: Temperature (°C)")
            logger.info("- feels_like: Feels like temperature (°C)")
            logger.info("- humidity: Relative humidity (%)")
            logger.info("- pressure: Atmospheric pressure (hPa)")
            logger.info("- wind_speed: Wind speed (m/s)")
            logger.info("- wind_direction: Wind direction (degrees)")
            logger.info("- clouds: Cloud cover (%)")
            logger.info("\nFiles saved:")
            logger.info(f"- Raw data: {raw_file}")
            logger.info(f"- Processed CSV: {csv_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {str(e)}")
            raise

def main():
    """Main function to fetch weather data."""
    try:
        # Initialize API client
        weather = WeatherAPI()
        
        # Fetch weather data
        weather.fetch_weather_data()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
