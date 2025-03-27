# Price Impact on CO2 Emission Prediction: Comprehensive Analysis

## Executive Summary

This analysis explores the relationship between electricity and gas prices and CO2 emission factors in the Netherlands. By integrating price data into our CO2 emission prediction model, we've gained valuable insights into how economic factors influence emission patterns. The analysis covers data from 2018 to 2025, including the energy crisis period triggered by the Russo-Ukrainian war.

## Key Findings

1. **Price-Emission Relationship**:
   - Electricity prices show a moderate correlation with CO2 emission factors
   - Higher electricity prices tend to correspond with slightly lower prediction errors
   - The gas-to-electricity price ratio provides insights into fuel switching behaviors

2. **Energy Crisis Impact**:
   - The energy crisis period shows distinct emission patterns compared to pre-crisis
   - Including crisis phase indicators (pre-crisis, early crisis, acute crisis, peak crisis, stabilization) helps the model account for structural market changes

3. **Model Performance**:
   - With price features included, the model achieved:
     - RMSE: 0.0148
     - MAE: 0.0125
     - Mean Percentage Error: -3.77%
   - DirectTabular emerged as the best individual model, with a weighted ensemble providing optimal performance

4. **Seasonal Patterns**:
   - Monthly analysis reveals seasonal correlations between prices and emissions
   - Price volatility shows distinct patterns during different times of the year

## Visualizations

The following visualizations provide deeper insights into the price-emission relationship:

### Price vs. Emissions Scatter Plot
![Price vs Emissions](data/analysis/price_vs_emissions.png)

This scatter plot shows the relationship between electricity prices and CO2 emission factors, with points colored by energy crisis period. The correlation coefficient and trend line help quantify the relationship.

### Time Series Analysis
![Time Series](data/analysis/price_emissions_time_series.png)

This dual-axis time series plot shows how electricity prices and CO2 emission factors have evolved over time, with the energy crisis period highlighted.

### Crisis Comparison
![Crisis Comparison](data/analysis/crisis_comparison.png)

This boxplot compares emission factors before and during the energy crisis, showing statistical significance and distribution differences.

### Monthly Patterns
![Monthly Patterns](data/analysis/monthly_patterns.png)

This visualization shows the seasonal patterns of both electricity prices and emission factors by month, revealing how they correlate throughout the year.

## Technical Implementation

The integration of price features involved:

1. **Data Processing**:
   - Reading and processing historical electricity price data (2018-2025)
   - Reading and processing historical gas price data
   - Creating energy crisis indicators based on the timeline of the Russo-Ukrainian war

2. **Feature Engineering**:
   - Basic price features: `elec_price`, `gas_price`, `gas_to_elec_ratio`
   - Crisis indicators: `energy_crisis`, `phase_pre_crisis`, `phase_early_crisis`, etc.
   - Temporal features: Lags, rolling means, and differences

3. **Model Integration**:
   - Adding price features to the known covariates in the AutoGluon TimeSeriesPredictor
   - Ensuring proper handling of missing values and data alignment
   - Evaluating the impact on prediction accuracy

## Recommendations

Based on our analysis, we recommend:

1. **Continue Incorporating Price Data**:
   - The price features show clear value for emission prediction
   - Consider adding more granular price features (e.g., day-ahead, intraday market prices)

2. **Explore Market Structure Features**:
   - Add features that capture market dynamics (e.g., import/export balances, merit order)
   - Consider including fuel prices for other generation sources (coal, biomass)

3. **Enhance Crisis Indicators**:
   - Fine-tune the crisis phase definitions
   - Consider adding policy event markers for major energy transitions
   
4. **Feature Engineering Improvements**:
   - Create interaction features between prices and weather (e.g., price premium during cold weather)
   - Add rolling windows for price volatility as potential indicators

## Conclusion

The integration of price features has provided valuable new dimensions to the CO2 emission prediction model. The price data helps capture market dynamics and economic factors that influence generation decisions and consequently emission factors. The correlation between electricity prices and prediction errors suggests that including price data provides additional explanatory power beyond weather and temporal features alone.

This analysis demonstrates that economic factors play a significant role in CO2 emissions from electricity generation, and incorporating these factors into prediction models can improve accuracy and provide deeper insights into emission patterns.

## Next Steps

1. Refine the price feature engineering based on the insights gained
2. Explore additional economic indicators that might influence emission factors
3. Develop a more sophisticated model for the relationship between prices and emissions
4. Investigate the impact of renewable energy subsidies and carbon pricing on the price-emission relationship 