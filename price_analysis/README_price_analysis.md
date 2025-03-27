# Price Analysis for CO2 Emission Factor Prediction

This project integrates electricity and gas price data into the CO2 emission factor prediction model for the Netherlands. The analysis explores the relationship between energy prices and CO2 emissions, with a particular focus on the impact of the energy crisis triggered by the Russo-Ukrainian war.

## Overview

The integration of price data provides valuable economic context to the CO2 emission prediction model, capturing market dynamics that influence generation decisions and consequently emission factors. This analysis demonstrates that economic factors play a significant role in CO2 emissions from electricity generation.

## Files and Structure

- `price_analysis.py`: Core module for reading, processing, and analyzing price data
- `visualize_price_impact.py`: Script for visualizing the relationship between prices and emissions
- `run_price_analysis_pipeline.py`: End-to-end pipeline that runs the entire analysis workflow
- `price_impact_summary.md`: Comprehensive summary of findings and recommendations
- `data/price/`: Directory containing electricity price data files
- `data/gas/`: Directory containing gas price data files
- `data/analysis/`: Directory containing generated visualizations and analysis outputs
- `data/price_features.csv`: Processed price features ready for model integration

## Key Features

1. **Price Data Processing**:
   - Reading and cleaning electricity price data (2018-2025)
   - Reading and cleaning gas price data
   - Creating energy crisis indicators based on the timeline of the Russo-Ukrainian war

2. **Feature Engineering**:
   - Basic price features: `elec_price`, `gas_price`, `gas_to_elec_ratio`
   - Crisis indicators: `energy_crisis`, `phase_pre_crisis`, `phase_early_crisis`, etc.
   - Temporal features: Lags, rolling means, and differences

3. **Visualization and Analysis**:
   - Price vs. emissions scatter plots
   - Time series analysis of prices and emissions
   - Crisis comparison analysis
   - Monthly and seasonal patterns
   - Correlation analysis by time of day and crisis period

4. **Model Integration**:
   - Integration with AutoGluon TimeSeriesPredictor
   - Addition of price features to known covariates
   - Evaluation of impact on prediction accuracy

## Usage

### Running the Pipeline

The complete analysis pipeline can be run with:

```bash
python run_price_analysis_pipeline.py
```

Optional arguments:
- `--price-dir`: Directory containing electricity price data (default: 'data/price')
- `--gas-dir`: Directory containing gas price data (default: 'data/gas')
- `--output-dir`: Directory to save analysis outputs (default: 'data/analysis')
- `--features-output`: Path to save the prepared price features (default: 'data/price_features.csv')
- `--skip-visualization`: Skip the visualization step
- `--skip-model-integration`: Skip the model integration step
- `--crisis-start`: Start date of the energy crisis (default: '2022-02-24')
- `--crisis-end`: End date of the energy crisis (default: None, meaning ongoing)

### Individual Components

You can also run individual components of the pipeline:

1. Process price data only:
```bash
python price_analysis.py
```

2. Generate visualizations:
```bash
python visualize_price_impact.py
```

3. Train the model with price features:
```bash
python train_model.py
```

4. Evaluate the model with price features:
```bash
python evaluate_model.py
```

## Key Findings

1. **Price-Emission Relationship**:
   - Electricity prices show a moderate correlation with CO2 emission factors
   - Higher electricity prices tend to correspond with slightly lower prediction errors
   - The gas-to-electricity price ratio provides insights into fuel switching behaviors

2. **Energy Crisis Impact**:
   - The energy crisis period shows distinct emission patterns compared to pre-crisis
   - Including crisis phase indicators helps the model account for structural market changes

3. **Seasonal Patterns**:
   - Monthly analysis reveals seasonal correlations between prices and emissions
   - Price volatility shows distinct patterns during different times of the year

## Visualizations

The analysis generates several visualizations:

- `price_vs_emissions.png`: Scatter plot showing the relationship between electricity prices and CO2 emission factors
- `price_emissions_time_series.png`: Time series plot of electricity prices and CO2 emission factors
- `crisis_comparison.png`: Boxplot comparing emission factors before and during the energy crisis
- `monthly_patterns.png`: Monthly patterns of electricity prices and CO2 emissions
- `price_trends.png`: Long-term trends in electricity and gas prices
- `price_volatility.png`: Price volatility analysis
- `price_correlation.png`: Correlation analysis between electricity and gas prices

## Model Integration

The price features have been integrated into the CO2 emission prediction model:

1. **train_model.py**: Updated to load and incorporate price features
2. **evaluate_model.py**: Updated to include price features in known_covariates

## Next Steps

1. Refine the price feature engineering based on the insights gained
2. Explore additional economic indicators that might influence emission factors
3. Develop a more sophisticated model for the relationship between prices and emissions
4. Investigate the impact of renewable energy subsidies and carbon pricing on the price-emission relationship

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- autogluon (for model integration)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Price data sourced from Dutch energy markets
- Analysis inspired by research on the impact of energy prices on carbon emissions 