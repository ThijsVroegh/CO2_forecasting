import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import numpy as np

from autogluon.timeseries import TimeSeriesPredictor
import read_ned
from train_model import gluonify, add_features
from config import MODEL_DIR, HISTORICAL_DIR, TRAINING_DAYS

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def load_test_data(use_full_context=False):
    """Load historical data as test set.
    
    Args:
        use_full_context: If True, use all available historical data as context.
                         If False, use only last 4 weeks as context.
    """
    ned_data = read_ned.read_all(HISTORICAL_DIR)
    ned_data.index = pd.to_datetime(ned_data.index)
    
    # Add alpha=0.05 for 95% confidence interval
    plot_pacf(ned_data.emissionfactor, lags=60, alpha=0.05)
    plt.savefig(MODEL_DIR / 'pacf_analysis.png')
    plt.close()
    
    if TRAINING_DAYS is not None:
        print(f"Using {TRAINING_DAYS} days of historical data")
        if use_full_context:
            # Use all available data within TRAINING_DAYS as context
            cutoff_date = ned_data.index[-1] - pd.Timedelta(days=TRAINING_DAYS)
            context_data = ned_data[
                (ned_data.index >= cutoff_date) & 
                (ned_data.index < ned_data.index[-7*24])
            ].copy()
            print("Using all available data as context within training period")
        else:
            # Use only last 4 weeks as context
            context_data = ned_data[-5*7*24:-7*24].copy()
            print("Using last 4 weeks as context")
    else:
        # Original behavior when TRAINING_DAYS is None
        if use_full_context:
            context_data = ned_data[:-7*24].copy()
            print("Using full historical data as context")
        else:
            context_data = ned_data[-5*7*24:-7*24].copy()
            print("Using last 4 weeks as context")
    
    # Add features to context data (including emission factor features)
    context_data = add_features(context_data)
    print("\nContext data features:")
    print("Emission factor features:", [col for col in context_data.columns if 'emissionfactor' in col])
    
    # test data
    test_data = ned_data[-7*24:].copy()
    test_data_with_features = add_features(test_data)       
    
    # Extract known covariates, i.e. all features that will be known during prediction/ forecasting 
    # horizon
    known_features = [
        # High importance features (based on plot)
        "volume_land-wind", "volume_sea-wind", "volume_sun",
        
        # Important temporal features
        "dayofweek", "is_weekend",
        "hour_sin", "hour_cos",  # Cyclical encoding of hour
        "dayofweek_sin", "dayofweek_cos",  # Cyclical encoding of day
        
        # Important weather features with their derived features
        # Wind features
        "wind_speed", "wind_speed_change_1h",
        "wind_speed_rolling_mean_24h",  # Explicitly include 24h
        "wind_speed_rolling_mean_168h", # Explicitly include 168h (1 week)
        
        # Temperature features (moderate importance)
        "temperature", "temperature_change_1h",
        "temperature_rolling_mean_24h",
        "temperature_rolling_mean_168h",
        
        # Solar/Radiation features (moderate importance)
        "solar_radiation", 
        "solar_radiation_rolling_mean_24h",
        "solar_radiation_rolling_mean_168h",
        "direct_radiation",
        "direct_radiation_rolling_mean_24h",
        "direct_radiation_rolling_mean_168h",
        
        # Keep some calendar features for special events
        "is_holiday", "is_summer_vacation"
    ]
    
    # Create covariates DataFrame with all known features
    covariates = test_data_with_features[known_features].copy()
    
    # Safety check for known_covariates
    covariate_cols = covariates.columns
    ef_features = [col for col in covariate_cols if 'emissionfactor' in col]
    if ef_features:
        raise ValueError(f"Found emission factor features in known_covariates: {ef_features}")
       
    print(f"Context period: {context_data.index[0]} to {context_data.index[-1]}")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Print feature counts by category
    print("\nFeature counts by category:")
    print(f"Renewable features: {sum(any(x in col for x in ['volume_sun', 'volume_land-wind', 'volume_sea-wind']) for col in covariates.columns)}")
    print(f"Temporal features: {sum(any(x in col for x in ['hour', 'dayofweek', 'weekend']) for col in covariates.columns)}")
    print(f"Wind features: {sum('wind' in col for col in covariates.columns)}")
    print(f"Temperature features: {sum('temperature' in col for col in covariates.columns)}")
    print(f"Solar/Radiation features: {sum(any(x in col for x in ['solar', 'radiation']) for col in covariates.columns)}")
    print(f"Holiday/Vacation features: {sum(any(x in col for x in ['holiday', 'vacation']) for col in covariates.columns)}")
    
    return context_data, test_data, covariates, test_data_with_features

def evaluate_forecast(use_full_context=False):
    """Generate and evaluate forecast for the test period.
    
    Evaluates model performance by:
    1. Making predictions using context data and known covariates
    2. Analyzing errors across different temporal patterns
    3. Visualizing predictions with renewable energy context
    
    Args:
        use_full_context: If True, use all historical data as context
                         If False, use only last 4 weeks
    
    Returns:
        tuple: (test_data, prediction) for further analysis if needed
    """
    # Load data
    context_data, test_data, known_covariates, test_data_with_features = load_test_data(use_full_context)
    
    # Verify features before prediction
    print("\nFeatures being used for prediction:")
    print("Context features:", context_data.columns.tolist())
    print("\nKnown covariates for prediction:", known_covariates.columns.tolist())
    
    # Verify all required features are present
    missing_features = set(known_covariates.columns) - set(test_data_with_features.columns)
    if missing_features:
        raise ValueError(f"Missing features in test data: {missing_features}")
    
    # Make prediction
    predictor = TimeSeriesPredictor.load(str(MODEL_DIR))
    gluon_context = gluonify(context_data)
    gluon_covariates = gluonify(known_covariates)
    prediction = predictor.predict(gluon_context, known_covariates = gluon_covariates)
    
    # Get prediction columns
    mean_col = [col for col in prediction.columns if 'mean' in col.lower()][0]
    quantile_10 = [col for col in prediction.columns if '0.1' in col][0]
    quantile_90 = [col for col in prediction.columns if '0.9' in col][0]
    
    # Calculate errors
    actual_values = test_data['emissionfactor'].values
    predicted_values = prediction[mean_col].values
    rmse = root_mean_squared_error(actual_values, predicted_values)
    
    # Create comprehensive error analysis table with all weather variables
    error_tbl = pd.DataFrame({
        'actual': actual_values,
        'predicted': predicted_values,
        'absolute_error': predicted_values - actual_values,
        'percentage_error': 100 * (predicted_values - actual_values) / actual_values,
        'actual_diff': test_data['emissionfactor'].diff().values,
        # Known covariates - renewable volumes
        'volume_sun': test_data['volume_sun'].values,
        'volume_land-wind': test_data['volume_land-wind'].values,
        'volume_sea-wind': test_data['volume_sea-wind'].values,
        # Known covariates - temporal features
        'hour': test_data_with_features['hour'].values,
        'dayofweek': test_data_with_features['dayofweek'].values,
        'is_weekend': test_data_with_features['is_weekend'].values,
        # Holiday and vacation features
        'is_holiday': test_data_with_features['is_holiday'].values,
        'is_school_vacation': test_data_with_features['is_school_vacation'].values,
        'is_summer_vacation': test_data_with_features['is_summer_vacation'].values,
        # Weather features
        'temperature': test_data_with_features['temperature'].values,
        'humidity': test_data_with_features['humidity'].values,
        'precipitation': test_data_with_features['precipitation'].values,
        'wind_speed': test_data_with_features['wind_speed'].values,
        'wind_direction': test_data_with_features['wind_direction'].values,
        'cloud_cover': test_data_with_features['cloud_cover'].values,
        'solar_radiation': test_data_with_features['solar_radiation'].values,
        'direct_radiation': test_data_with_features['direct_radiation'].values,
        'diffuse_radiation': test_data_with_features['diffuse_radiation'].values,
        'direct_normal_irradiance': test_data_with_features['direct_normal_irradiance'].values,
        'global_irradiance': test_data_with_features['global_irradiance'].values,
        'terrestrial_radiation': test_data_with_features['terrestrial_radiation'].values
    }, index=test_data.index)
    
    # Generate timestamp for all saved files
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    context_type = "full_context" if use_full_context else "4weeks_context"
    
    # Plot 1: Detailed Error Analysis
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
    ax1.plot(test_data.index, error_tbl['absolute_error'], 
            label='Absolute Error', color='gray', linestyle='--')
    ax1.set_ylabel('Emission Factor (kgCO2eq/kWh)')
    
    # Calculate additional metrics
    mae = mean_absolute_error(test_data['emissionfactor'].values, prediction[mean_col].values)
    mape = mean_absolute_percentage_error(test_data['emissionfactor'].values, prediction[mean_col].values) * 100
    r2 = r2_score(test_data['emissionfactor'].values, prediction[mean_col].values)
    
    ax1.set_title(f'Prediction Analysis (RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.1f}%, R²: {r2:.3f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()      
   
    # Middle panel: Renewable Energy
    ax2 = fig.add_subplot(gs[1])
    for col in ['volume_sun', 'volume_land-wind', 'volume_sea-wind']:
        ax2.plot(test_data.index, 
                error_tbl[col] / error_tbl[col].max(),
                label=f'{col} (normalized)', alpha=0.6)
    ax2.set_ylabel('Normalized Production')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Bottom panel: Error Patterns
    ax3 = fig.add_subplot(gs[2])
    # Plot hourly error pattern
    hourly_errors = error_tbl.groupby('hour')['absolute_error'].mean()
    ax3.plot(hourly_errors.index, hourly_errors.values, 
            label='Mean Hourly Error', color='purple')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(
        MODEL_DIR / f'detailed_analysis_{context_type}_{timestamp}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    # Save detailed error analysis
    error_analysis = pd.DataFrame({
        'Hourly_Error': error_tbl.groupby('hour')['absolute_error'].mean(),
        'Daily_Error': error_tbl.groupby('dayofweek')['absolute_error'].mean(),
        'Weekend_Error': error_tbl.groupby('is_weekend')['absolute_error'].mean(),
        'Hourly_RMSE': np.sqrt(error_tbl.groupby('hour')['absolute_error'].apply(lambda x: (x**2).mean())),
        'Daily_RMSE': np.sqrt(error_tbl.groupby('dayofweek')['absolute_error'].apply(lambda x: (x**2).mean())),
    })
    error_analysis.to_csv(MODEL_DIR / f'error_analysis_{context_type}_{timestamp}.csv')
    
    # Print comprehensive summary
    print("\nError Analysis Summary:")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Mean Absolute Error: {error_tbl['absolute_error'].abs().mean():.4f}")
    print(f"Mean Percentage Error: {error_tbl['percentage_error'].mean():.2f}%")
    print(f"\nTemporal Patterns:")
    print(f"Best Hour: {hourly_errors.idxmin()} ({hourly_errors.min():.4f})")
    print(f"Worst Hour: {hourly_errors.idxmax()} ({hourly_errors.max():.4f})")
    print(f"Weekday MAE: {error_tbl[error_tbl['is_weekend']==0]['absolute_error'].abs().mean():.4f}")
    print(f"Weekend MAE: {error_tbl[error_tbl['is_weekend']==1]['absolute_error'].abs().mean():.4f}")
    
    # Add temperature analysis if available
    if 'temperature' in error_tbl.columns and not error_tbl['temperature'].isna().all():
        print("\nTemperature Analysis:")
        
        # Create temperature bins with observed=True to handle the warning
        temp_bins = pd.qcut(error_tbl['temperature'], q=4, labels=[
            'Very Cold', 'Cold', 'Warm', 'Very Warm'
        ])
        
        # Calculate statistics separately
        temp_means = error_tbl.groupby(temp_bins, observed=True)['absolute_error'].mean()
        temp_stds = error_tbl.groupby(temp_bins, observed=True)['absolute_error'].std()
        temp_counts = error_tbl.groupby(temp_bins, observed=True)['absolute_error'].count()
        
        # Get temperature ranges
        temp_ranges = pd.qcut(error_tbl['temperature'], q=4, retbins=True)[1]
        
        print("\nError by temperature range:")
        for idx in temp_means.index:
            temp_range = f"{temp_ranges[list(temp_means.index).index(idx)]:.1f}°C to {temp_ranges[list(temp_means.index).index(idx)+1]:.1f}°C"
            print(f"  {idx:9} ({temp_range:20}): "
                  f"MAE = {temp_means[idx]:.4f} ± "
                  f"{temp_stds[idx]:.4f} "
                  f"(n={int(temp_counts[idx])})")
        
        # Add correlation analysis
        temp_corr = error_tbl[['temperature', 'absolute_error']].corr().iloc[0,1]
        print(f"\nCorrelation between temperature and absolute error: {temp_corr:.3f}")
    
    # Add difference analysis to the summary
    print("\nDifference Analysis:")
    print(f"Mean Absolute Diff: {error_tbl['actual_diff'].abs().mean():.4f}")
    corr = error_tbl[['actual_diff', 'absolute_error']].corr().iloc[0,1]
    print(f"Correlation between difference and error: {corr:.3f}")
    
    # Add holiday/vacation analysis
    print("\nHoliday Analysis:")
    print(f"Holiday MAE: {error_tbl[error_tbl['is_holiday']==1]['absolute_error'].abs().mean():.4f}")
    print(f"Non-Holiday MAE: {error_tbl[error_tbl['is_holiday']==0]['absolute_error'].abs().mean():.4f}")
    
    print("\nVacation Analysis:")
    print(f"School Vacation MAE: {error_tbl[error_tbl['is_school_vacation']==1]['absolute_error'].abs().mean():.4f}")
    print(f"Non-Vacation MAE: {error_tbl[error_tbl['is_school_vacation']==0]['absolute_error'].abs().mean():.4f}")
    print(f"Summer Vacation MAE: {error_tbl[error_tbl['is_summer_vacation']==1]['absolute_error'].abs().mean():.4f}")
    
    # Add comprehensive weather analysis to the summary
    print("\nWeather Feature Analysis:")
    weather_features = [
        'temperature', 'humidity', 'precipitation', 'wind_speed', 'cloud_cover',
        'solar_radiation', 'direct_radiation', 'diffuse_radiation',
        'direct_normal_irradiance', 'global_irradiance', 'terrestrial_radiation'
    ]
    
    for feature in weather_features:
        if feature in error_tbl.columns:
            print(f"\n{feature.title()} Analysis:")
            
            # Check if the feature has enough unique values for quartile binning
            unique_values = error_tbl[feature].nunique()
            
            if unique_values <= 1:
                print(f"  Warning: {feature} has only {unique_values} unique value(s)")
                print(f"  Single value: {error_tbl[feature].iloc[0]}")
                continue
                
            try:
                # Try to create quartile bins, handling duplicates
                if error_tbl[feature].nunique() < 4:
                    # Not enough unique values for quartiles, use unique values
                    bins = pd.cut(error_tbl[feature], 
                                bins=unique_values,
                                labels=[f'{feature} - B{i+1}' for i in range(unique_values)])
                else:
                    # Use qcut with duplicates='drop' to handle duplicate edges
                    bins = pd.qcut(error_tbl[feature], 
                                 q=4, 
                                 labels=[f'{feature} - Q{i+1}' for i in range(4)],
                                 duplicates='drop')
                
                # Calculate statistics with observed=True
                bin_stats = error_tbl.groupby(bins, observed=True)['absolute_error'].agg(['mean', 'std', 'count'])
                
                # Get actual value ranges for each bin
                value_ranges = {}
                for bin_label in bin_stats.index:
                    mask = bins == bin_label
                    values = error_tbl[feature][mask]
                    value_ranges[bin_label] = (values.min(), values.max())
                
                # Print statistics with value ranges
                for idx in bin_stats.index:
                    value_range = value_ranges[idx]
                    print(f"  {idx:30} ({value_range[0]:.2f} to {value_range[1]:.2f}): "
                          f"MAE = {bin_stats.loc[idx, 'mean']:.4f} ± "
                          f"{bin_stats.loc[idx, 'std']:.4f} "
                          f"(n={int(bin_stats.loc[idx, 'count'])})")
                
                # Add correlation analysis
                corr = error_tbl[[feature, 'absolute_error']].corr().iloc[0,1]
                print(f"  Correlation with absolute error: {corr:.3f}")
                
                # Add basic statistics
                print(f"  Feature statistics:")
                print(f"    Mean: {error_tbl[feature].mean():.2f}")
                print(f"    Std: {error_tbl[feature].std():.2f}")
                print(f"    Min: {error_tbl[feature].min():.2f}")
                print(f"    Max: {error_tbl[feature].max():.2f}")
                print(f"    Zero values: {(error_tbl[feature] == 0).sum()} ({(error_tbl[feature] == 0).mean()*100:.1f}%)")
                
            except Exception as e:
                print(f"  Warning: Could not analyze {feature}: {str(e)}")
                print(f"  Feature statistics:")
                print(f"    Unique values: {unique_values}")
                print(f"    Value counts:\n{error_tbl[feature].value_counts().head()}")
    
    return test_data, prediction

if __name__ == "__main__":
    try:
        # First evaluate with 4 weeks context
        print("\n=== Evaluating with 4 weeks context ===")
        test_data, prediction = evaluate_forecast(use_full_context=False)
        
        # Then evaluate with full historical context
        print("\n=== Evaluating with full historical context ===")
        test_data, prediction = evaluate_forecast(use_full_context=True)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}") 
        
