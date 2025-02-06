import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

from config import FORECAST_DIR

def load_latest_forecasts():
    """Load the most recent forecast files."""
    # Get most recent files
    
    # contain NED's renewable energy forecasts used as input data for predictions
    forecast_files = list(FORECAST_DIR.glob("forecast_*.csv"))
    
    # Contain emission factor predictions and confidence intervals for the next 7 days. 
    # Used for operational purposes: real-time forecasting.
    prediction_files = list(FORECAST_DIR.glob("prediction_*.csv"))
    
    if not forecast_files or not prediction_files:
        raise FileNotFoundError("No forecast files found")
    
    # Get most recent files
    latest_forecast = max(forecast_files, key=lambda x: x.stat().st_mtime)
    latest_prediction = max(prediction_files, key=lambda x: x.stat().st_mtime)
    
    # Load forecast data
    forecast_df = pd.read_csv(latest_forecast, index_col=0)
    forecast_df.index = pd.to_datetime(forecast_df.index)
    forecast_df = forecast_df.sort_index()
    
    # Load prediction data and sync its index with forecast data
    prediction_df = pd.read_csv(latest_prediction, index_col=0)
    prediction_df.index = forecast_df.index  # Use same index as forecast
    prediction_df = prediction_df.sort_index()
    
    # Print debug info
    print("Forecast period:", forecast_df.index[0], "to", forecast_df.index[-1])
    print("Prediction period:", prediction_df.index[0], "to", prediction_df.index[-1])
    
    return forecast_df, prediction_df

def plot_forecasts(forecast_df, prediction_df):
    """Create plots of the forecasts."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])
    
    # Plot emission factor forecast (top)
    mean_col = [col for col in prediction_df.columns if 'mean' in col.lower()][0]
    quantile_10 = [col for col in prediction_df.columns if '0.1' in col][0]
    quantile_90 = [col for col in prediction_df.columns if '0.9' in col][0]
    
    ax1.fill_between(prediction_df.index, 
                     prediction_df[quantile_10], 
                     prediction_df[quantile_90],
                     alpha=0.3, color='blue', label='Forecast (10-90%)')
    ax1.plot(prediction_df.index, prediction_df[mean_col], 
             color='blue', label='Forecast (mean)', linewidth=2)
    
    ax1.set_ylabel('Emission factor\n(kgCO2eq/kWh)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates for top plot
    ax1.tick_params(axis='x', rotation=45)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Plot renewable production forecast (bottom)
    ax2.fill_between(forecast_df.index, 0, 
                     forecast_df['volume_sea-wind']/1e6,
                     label='sea-wind', color='darkblue')
    ax2.fill_between(forecast_df.index, 
                     forecast_df['volume_sea-wind']/1e6,
                     (forecast_df['volume_sea-wind'] + forecast_df['volume_land-wind'])/1e6,
                     label='land-wind', color='lightblue')
    ax2.fill_between(forecast_df.index, 
                     (forecast_df['volume_sea-wind'] + forecast_df['volume_land-wind'])/1e6,
                     (forecast_df['volume_sea-wind'] + forecast_df['volume_land-wind'] + 
                      forecast_df['volume_sun'])/1e6,
                     label='solar', color='orange')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Green production\n(MW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates for bottom plot
    ax2.tick_params(axis='x', rotation=45)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Adjust layout to prevent date labels from being cut off
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    plt.savefig(FORECAST_DIR / f'forecast_plot_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        forecast_df, prediction_df = load_latest_forecasts()
        plot_forecasts(forecast_df, prediction_df)
        print("Forecast plots created successfully!")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")

if __name__ == "__main__":
    main() 