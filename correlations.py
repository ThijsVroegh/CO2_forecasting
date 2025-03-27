import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# this script is used to analyze the correlation between the historical data on 
# the electricity demand and the temperature data. 

# Read the CSV files
historical_df = pd.read_csv('data/downloaded/historical_combined_20250207.csv')
temperatures_df = pd.read_csv('data/knmi_data/processed_temperatures.csv')

# Convert datetime columns to datetime type
historical_df['time'] = pd.to_datetime(historical_df['time'])
temperatures_df['datetime'] = pd.to_datetime(temperatures_df['datetime'])

# Rename 'time' column to match with temperatures_df
historical_df = historical_df.rename(columns={'time': 'datetime'})

# Merge the dataframes on datetime
merged_df = pd.merge(historical_df, temperatures_df, on='datetime', how='inner')

# Filter data starting from 2021-01-01 01:00:00
start_date = '2021-01-01 01:00:00'
filtered_df = merged_df[merged_df['datetime'] >= start_date]

# Set datetime as index
filtered_df.set_index('datetime', inplace=True)

# Calculate total wind
filtered_df['volume_total_wind'] = filtered_df['volume_land-wind'] + filtered_df['volume_sea-wind']

# Add time-based features
filtered_df['hour'] = filtered_df.index.hour
filtered_df['day_of_week'] = filtered_df.index.dayofweek
filtered_df['month'] = filtered_df.index.month

# Variables to analyze
vars_to_check = ['temperature_celsius', 'emissionfactor', 'volume_sun', 'volume_total_wind']

# Create stationary versions of the variables (using first differences)
stationary_df = pd.DataFrame(index=filtered_df.index)
for var in vars_to_check:
    stationary_df[var] = filtered_df[var].diff()
    
# Remove NaN values created by differencing
stationary_df = stationary_df.dropna()

# Add time features to stationary data
stationary_df['hour'] = filtered_df['hour'][1:]  # Skip first row due to differencing
stationary_df['day_of_week'] = filtered_df['day_of_week'][1:]
stationary_df['month'] = filtered_df['month'][1:]

# Analyze hourly patterns of changes
plt.figure(figsize=(15, 10))
for i, var in enumerate(vars_to_check, 1):
    plt.subplot(2, 2, i)
    hourly_means = stationary_df.groupby('hour')[var].mean()
    plt.plot(hourly_means.index, hourly_means.values)
    plt.title(f'Average Hourly Changes in {var}')
    plt.xlabel('Hour of Day')
    plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate correlations for each hour using stationary data
hourly_correlations = {}
for hour in range(24):
    hour_data = stationary_df[stationary_df['hour'] == hour]
    corr = hour_data['temperature_celsius'].corr(hour_data['emissionfactor'])
    hourly_correlations[hour] = corr

# Plot hourly correlations
plt.figure(figsize=(12, 6))
plt.plot(list(hourly_correlations.keys()), list(hourly_correlations.values()), marker='o')
plt.title('Temperature-Emission Change Correlation by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.show()

# Calculate lagged correlations (up to 24 hours) using stationary data
max_lag = 24
lag_correlations = []
for lag in range(max_lag + 1):
    temp_lagged = stationary_df['temperature_celsius'].shift(lag)
    corr = temp_lagged.corr(stationary_df['emissionfactor'])
    lag_correlations.append(corr)

# Plot lagged correlations
plt.figure(figsize=(12, 6))
plt.plot(range(max_lag + 1), lag_correlations, marker='o')
plt.title('Temperature-Emission Change Correlation by Lag (Hours)')
plt.xlabel('Lag (Hours)')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.show()

# Print summary statistics
print("\nCorrelation Summary (using changes/differences):")
print("\nOverall correlation:", 
      stationary_df['temperature_celsius'].corr(stationary_df['emissionfactor']))

print("\nStrongest hourly correlation:", 
      max(hourly_correlations.items(), key=lambda x: abs(x[1])))

print("\nStrongest lag correlation:", 
      max(enumerate(lag_correlations), key=lambda x: abs(x[1])))

# Create correlation matrix for all variables
correlation_matrix = stationary_df[vars_to_check].corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            center=0,
            square=True,
            fmt='.2f')
plt.title('Correlations between Changes in Variables')
plt.tight_layout()
plt.show()
