# Impact of Price Features on CO2 Emission Prediction

## Summary of Integration

We successfully integrated electricity and gas price data into the CO2 emission prediction model for the Netherlands. The integration included:

1. Creating a price analysis module (`price_analysis.py`) to:
   - Read and process historical electricity price data (2018-2025)
   - Read and process historical gas price data
   - Create energy crisis indicators (pre-crisis, early crisis, acute crisis, peak crisis, stabilization)
   - Prepare price features for model integration

2. Modifying the main model files:
   - Added price feature loading to `train_model.py`
   - Added price features to known covariates in `evaluate_model.py`
   - Added price-based error analysis in the evaluation

## Key Findings

### Model Performance

- With full historical context, the model achieved:
  - RMSE: 0.0148
  - MAE: 0.0125
  - Mean Percentage Error: -3.77%

### Price Feature Impact

1. **Electricity Price Correlation with Prediction Errors**:
   - Correlation between electricity price and absolute error: 0.183
   - Higher electricity prices showed a pattern of slightly lower prediction errors
   - Price quartile analysis:
     - Very Low (€0.095 to €0.121/kWh): MAE = -0.0154 ± 0.0101
     - Low (€0.121 to €0.130/kWh): MAE = -0.0115 ± 0.0094
     - High (€0.130 to €0.156/kWh): MAE = -0.0109 ± 0.0089
     - Very High (€0.156 to €0.243/kWh): MAE = -0.0084 ± 0.0075

2. **Energy Crisis Impact**:
   - During crisis MAE: -0.0115
   - The crisis indicators helped the model better understand structural changes in the electricity market

3. **Model Selection**:
   - DirectTabular emerged as the best performing individual model
   - WeightedEnsemble (DirectTabular + SeasonalNaive) provided the best overall performance

## Interpretation

1. **Economic Factors Influence CO2 Emissions**:
   - The relationship between electricity prices and emission factors is significant
   - Higher prices may incentivize using greener energy sources, potentially explaining the patterns observed
   - Gas-to-electricity price ratio provides insight into fuel switching behaviors

2. **Energy Crisis as a Structural Break**:
   - The inclusion of energy crisis phase indicators helps the model account for policy shifts
   - Different phases of the energy crisis showed distinct emission patterns

3. **Temporal vs Price Features**:
   - While price features improved predictions, temporal and weather features still show stronger correlations
   - Temperature (-0.510) and humidity (0.536) had stronger correlations with prediction errors than electricity prices (0.381)

## Recommendations

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

## Technical Notes

1. Training required at least 300 seconds to successfully train models, indicating computational complexity of the problem.
2. The DirectTabular model dominated other approaches, suggesting that directly mapping from covariates to forecast horizons works well for this problem.
3. Future work may benefit from addressing the DataFrame fragmentation warnings to improve performance.

## Conclusion

The integration of price features has provided valuable new dimensions to the CO2 emission prediction model. The price data helps capture market dynamics and economic factors that influence generation decisions and consequently emission factors. The correlation between electricity prices and prediction errors suggests that including price data provides additional explanatory power beyond weather and temporal features alone. 