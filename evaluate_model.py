import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import numpy as np

from autogluon.timeseries import TimeSeriesPredictor
import read_ned
from train_model import gluonify, add_features
from config import MODEL_DIR, HISTORICAL_DIR, TRAINING_DAYS

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error,mean_absolute_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def backtest_evaluate(context_data, prediction_length=7*24, num_windows=3):
    """Perform backtesting evaluation using multiple windows.
    
    Args:
        context_data: DataFrame containing the full historical data
        prediction_length: Number of time steps to predict (default: 1 week of hourly data)
        num_windows: Number of validation windows to use
        
    Returns:
        dict: Aggregated metrics across all windows
    """
    # Calculate window parameters
    total_length = len(context_data)
    window_length = prediction_length
    
    # Initialize storage for metrics across windows
    window_metrics = []
    all_predictions = []
    all_actuals = []
    
    # Generate windows from newest to oldest
    for window_idx in range(num_windows):
        print(f"\n=== Evaluating Window {window_idx + 1}/{num_windows} ===")
        
        # Calculate window boundaries
        end_idx = total_length - (window_idx * window_length)
        start_idx = end_idx - window_length
        
        # Extract test data for this window
        test_data = context_data[start_idx:end_idx].copy()
        
        # Use data before the test window as context
        if window_idx == num_windows - 1:
            # For the last (oldest) window, use all available prior data
            context_window = context_data[:start_idx].copy()
        else:
            # For other windows, use 4 weeks of context
            context_window = context_data[max(0, start_idx-4*7*24):start_idx].copy()
            
        # Add features to both context and test data
        context_window = add_features(context_window)
        test_data_with_features = add_features(test_data)
        
        # Extract known covariates for test period
        known_features = [
            'volume_sun', 'volume_land-wind', 'volume_sea-wind',
            'hour', 'dayofweek', 'month', 'is_weekend',
            'day_of_year_sin', 'day_of_year_cos',
            'week_of_year_sin', 'week_of_year_cos',
            'hour_sin', 'hour_cos',
            'dayofweek_sin', 'dayofweek_cos',
            'month_sin', 'month_cos',
            'quarter_sin', 'quarter_cos',
            'temperature_celsius', 'temperature_change_1h', 'temperature_change_24h',
            *[f'temperature_lag_{lag}h' for lag in [1, 8, 9, 10, 20, 21, 22, 23, 24]],
            *[f'temperature_rolling_mean_{window}h' for window in [24, 168]],
            'is_holiday', 'is_holiday_adjacent',
            'is_school_vacation', 'is_summer_vacation', 'vacation_type'
        ]
        
        covariates = test_data_with_features[known_features].copy()
        
        # Make prediction
        predictor = TimeSeriesPredictor.load(str(MODEL_DIR))
        gluon_context = gluonify(context_window)
        gluon_covariates = gluonify(covariates)
        prediction = predictor.predict(gluon_context, known_covariates=gluon_covariates)
        
        # Store predictions and actuals
        mean_col = [col for col in prediction.columns if 'mean' in col.lower()][0]
        all_predictions.extend(prediction[mean_col].values)
        all_actuals.extend(test_data['emissionfactor'].values)
        
        # Calculate metrics for this window
        window_metrics.append({
            'window': window_idx + 1,
            'start_date': test_data.index[0],
            'end_date': test_data.index[-1],
            'rmse': root_mean_squared_error(
                test_data['emissionfactor'].values,
                prediction[mean_col].values
            ),
            'mae': mean_absolute_error(
                test_data['emissionfactor'].values,
                prediction[mean_col].values
            ),
            'r2': r2_score(
                test_data['emissionfactor'].values,
                prediction[mean_col].values
            )
        })
        
        # Generate visualization for this window
        plot_window_analysis(
            test_data,
            prediction,
            test_data_with_features,
            window_idx + 1,
            window_metrics[-1]['rmse']
        )
    
    # Convert window metrics to DataFrame for aggregation
    metrics_df = pd.DataFrame(window_metrics)
    
    # Calculate mean and median metrics across windows
    mean_metrics = metrics_df[['rmse', 'mae', 'r2']].mean()
    median_metrics = metrics_df[['rmse', 'mae', 'r2']].median()
    
    # Calculate overall metrics across all predictions
    aggregate_metrics = {
        'overall_rmse': root_mean_squared_error(all_actuals, all_predictions),
        'overall_mae': mean_absolute_error(all_actuals, all_predictions),
        'overall_r2': r2_score(all_actuals, all_predictions),
        'mean_window_rmse': mean_metrics['rmse'],
        'mean_window_mae': mean_metrics['mae'],
        'mean_window_r2': mean_metrics['r2'],
        'median_window_rmse': median_metrics['rmse'],
        'median_window_mae': median_metrics['mae'],
        'median_window_r2': median_metrics['r2'],
        'window_metrics': window_metrics
    }
    
    # Save metrics to file
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')    
    
    # Create a dictionary with all metrics
    # Save aggregate metrics
    aggregate_df = pd.DataFrame({
        'metric': [
            'overall_rmse', 'overall_mae', 'overall_r2',
            'mean_window_rmse', 'mean_window_mae', 'mean_window_r2',
            'median_window_rmse', 'median_window_mae', 'median_window_r2'
        ],
        'value': [
            aggregate_metrics['overall_rmse'],
            aggregate_metrics['overall_mae'],
            aggregate_metrics['overall_r2'],
            aggregate_metrics['mean_window_rmse'],
            aggregate_metrics['mean_window_mae'],
            aggregate_metrics['mean_window_r2'],
            aggregate_metrics['median_window_rmse'],
            aggregate_metrics['median_window_mae'],
            aggregate_metrics['median_window_r2']
        ]
    })
    
    # Save individual window metrics
    window_metrics_df = pd.DataFrame(window_metrics)
    
    # Save both DataFrames to CSV
    metrics_base_path = MODEL_DIR / f'backtest_metrics_{timestamp}'
    aggregate_df.to_csv(f'{metrics_base_path}_aggregate.csv', index=False)
    window_metrics_df.to_csv(f'{metrics_base_path}_windows.csv', index=False)
    
    # Print summary
    print("\n=== Backtesting Summary ===")
    print(f"Overall RMSE: {aggregate_metrics['overall_rmse']:.4f}")
    print(f"Overall MAE: {aggregate_metrics['overall_mae']:.4f}")
    print(f"Overall R²: {aggregate_metrics['overall_r2']:.4f}")
    print("\nWindow-specific metrics:")
    for metrics in window_metrics:
        print(f"\nWindow {metrics['window']}:")
        print(f"  Period: {metrics['start_date']} to {metrics['end_date']}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    return aggregate_metrics

def plot_window_analysis(test_data, prediction, test_data_with_features, window_num, rmse):
    """Generate detailed analysis plot for a specific window."""
    # Get prediction columns
    mean_col = [col for col in prediction.columns if 'mean' in col.lower()][0]
    quantile_10 = [col for col in prediction.columns if '0.1' in col][0]
    quantile_90 = [col for col in prediction.columns if '0.9' in col][0]
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Top panel: Predictions and Errors
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(test_data.index, test_data['emissionfactor'], 
            label='Actual', color='blue', linewidth=2)
    ax1.plot(test_data.index, prediction[mean_col], 
            label='Predicted', color='red', linewidth=2)
    ax1.fill_between(test_data.index, 
                    prediction[quantile_10], 
                    prediction[quantile_90],
                    alpha=0.2, color='red', label='90% Confidence')
    ax1.set_ylabel('Emission Factor (kgCO2eq/kWh)')
    # Calculate additional metrics
    mae = mean_absolute_error(test_data['emissionfactor'].values, prediction[mean_col].values)
    mape = mean_absolute_percentage_error(test_data['emissionfactor'].values, prediction[mean_col].values) * 100  # Convert to percentage
    r2 = r2_score(test_data['emissionfactor'].values, prediction[mean_col].values)
    
    ax1.set_title(f'Window {window_num} Analysis\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.1f}%, R²: {r2:.3f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Middle panel: Renewable Energy
    ax2 = fig.add_subplot(gs[1])
    for col in ['volume_sun', 'volume_land-wind', 'volume_sea-wind']:
        ax2.plot(test_data.index, 
                test_data[col] / test_data[col].max(),
                label=f'{col} (normalized)', alpha=0.6)
    ax2.set_ylabel('Normalized Production')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Bottom panel: Error Patterns
    ax3 = fig.add_subplot(gs[2])
    absolute_errors = abs(test_data['emissionfactor'].values - prediction[mean_col].values)
    hourly_errors = pd.DataFrame({
        'hour': test_data_with_features['hour'],
        'error': absolute_errors
    }).groupby('hour')['error'].mean()
    
    ax3.plot(hourly_errors.index, hourly_errors.values, 
            label='Mean Hourly Error', color='purple')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    plt.savefig(
        MODEL_DIR / f'window_{window_num}_analysis_{timestamp}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()

if __name__ == "__main__":
    try:
        # Load the full context data
        ned_data = read_ned.read_all(HISTORICAL_DIR)
        ned_data.index = pd.to_datetime(ned_data.index)
        
        # Perform backtesting evaluation
        metrics = backtest_evaluate(ned_data, prediction_length=7*24, num_windows=3)
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")