import datetime
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor

import retrieve_ned
from train_model import gluonify, add_features
from config import MODEL_DIR, FORECAST_DIR, DOWNLOADED_DIR



def predict():
    """Generate emission factor forecasts using the trained model.
    
    This function uses a two-step approach:
    1. Recent Context: Gets the last 4 weeks of data to understand current trends
       (While the model is trained on much more data, it only needs recent context
       for making predictions, similar to how weather forecasts use recent conditions)
    2. Future Information: Gets NED's renewable energy forecasts for the next 7 days
    
    The model combines:
    - Long-term patterns (learned during training from years of data)
    - Recent trends (from 4 weeks of context)
    - Known future renewable production (from NED forecasts)
    - Lagged features (1h, 2h, 3h, 24h, 48h, 168h)
    - Rolling means (24h, 168h)
    to predict emission factors.
    
    The function saves three files with current timestamp:
    1. FORECAST_DIR/prediction_{date}.csv: 
       The predicted emission factors with confidence intervals
    2. DOWNLOADED_DIR/runup_data_{date}.csv: 
       The historical context data used (last RUNUP_PERIOD days)
    3. DOWNLOADED_DIR/ned_forecast_{date}.csv: 
       NED's renewable energy forecasts used as covariates
    
    Returns:
        None       
        
    See Also:
        retrieve_ned.RUNUP_PERIOD: Constant defining how many days of context to use
        retrieve_ned.get_historical_data(): Function used for model training
    """
    
    # Get data
    
    # 'runup_data' contains historical data used as context for the last 4 weeks. It 
    # includes renewable energy volumes and emission factors.
    runup_data = retrieve_ned.get_runup_data()
    
    # 'forecast_data' contains NED's renewable energy forecasts used as known input
    # data in the forecasting horizon used for Co2 predictions for the next 7 days. 
    forecast_data = retrieve_ned.get_current_forecast()
    
    # Ensure datetime index
    runup_data.index = pd.to_datetime(runup_data.index)
    forecast_data.index = pd.to_datetime(forecast_data.index)
    
    # Add all features to both datasets
    runup_data_with_features = add_features(runup_data)
    print("Runup data features after adding features:", runup_data_with_features.columns.tolist())
    
    forecast_data_with_features = add_features(forecast_data)
    print("Forecast data features after adding features:", forecast_data_with_features.columns.tolist())
    
    # Debug: print features in forecast data
    print("\nFeatures in forecast data:")
    print(forecast_data_with_features.columns.tolist())
    print("\nFirst few rows of forecast data:")
    print(forecast_data_with_features.head())
    
    # Convert to AutoGluon format
    gluon_runup = gluonify(runup_data_with_features)
    gluon_forecast = gluonify(forecast_data_with_features)
    
    # Load model and make prediction
    predictor = TimeSeriesPredictor.load(str(MODEL_DIR))
    prediction = predictor.predict(gluon_runup, known_covariates=gluon_forecast)
            
    # Save results
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    prediction.to_csv(FORECAST_DIR / f"prediction_{date}.csv")
    gluon_runup.to_csv(DOWNLOADED_DIR / f"runup_data_{date}.csv")
    gluon_forecast.to_csv(DOWNLOADED_DIR / f"ned_forecast_{date}.csv")
    
    print("\nPrediction completed successfully!")
    print(f"Context data shape: {gluon_runup.shape}")
    print(f"Forecast data shape: {gluon_forecast.shape}")
    print("\nFeature counts:")
    print(f"Temporal features: {sum(col in ['hour', 'dayofweek', 'month', 'is_weekend'] for col in gluon_runup.columns)}")
    print(f"Lag features: {sum('lag' in col for col in gluon_runup.columns)}")
    print(f"Rolling mean features: {sum('rolling_mean' in col for col in gluon_runup.columns)}")

if __name__ == "__main__":
    predict()
