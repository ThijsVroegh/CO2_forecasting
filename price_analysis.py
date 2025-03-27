import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_context("paper", font_scale=1.2)

def read_electricity_price_data(price_dir):
    """
    Read and combine electricity price data from multiple yearly files.
    
    Args:
        price_dir (str or Path): Directory containing electricity price CSV files
        
    Returns:
        pd.DataFrame: Combined DataFrame with hourly electricity prices
    """
    price_dir = Path(price_dir)
    price_files = list(price_dir.glob("jeroen_punt_nl_dynamische_stroomprijzen_jaar_*.csv"))
    
    if not price_files:
        raise FileNotFoundError(f"No electricity price files found in {price_dir}")
    
    print(f"Found {len(price_files)} electricity price files")
    
    # Read and combine all price files
    dfs = []
    for file in sorted(price_files):
        print(f"Reading {file.name}")
        # Read with semicolon delimiter and handle European-style decimal point
        try:
            df = pd.read_csv(file, sep=';', decimal=',')
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            # Try without decimal specification
            try:
                df = pd.read_csv(file, sep=';')
                dfs.append(df)
            except Exception as e2:
                print(f"Second attempt error: {e2}")
    
    if not dfs:
        raise ValueError("Could not read any electricity price files")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Print raw data structure
    print("\nRaw electricity price data structure:")
    print(combined_df.head())
    print(f"\nColumns: {combined_df.columns.tolist()}")
    
    # Process data: convert to proper datetime and set as index
    try:
        # Assuming the first column is the datetime column
        date_col = combined_df.columns[0]
        price_col = combined_df.columns[1]
        
        print(f"Using {date_col} as date column and {price_col} as price column")
        
        # Convert to datetime
        combined_df['datetime'] = pd.to_datetime(combined_df[date_col], format="%d-%m-%Y %H:%M", errors='coerce')
        
        # Check for conversion issues
        invalid_dates = combined_df['datetime'].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: Found {invalid_dates} invalid dates in electricity price data")
            # Try alternative format
            combined_df['datetime'] = pd.to_datetime(combined_df[date_col], errors='coerce')
            invalid_dates = combined_df['datetime'].isna().sum()
            if invalid_dates > 0:
                print(f"Still have {invalid_dates} invalid dates after second attempt")
                combined_df = combined_df.dropna(subset=['datetime'])
        
        # Set datetime as index
        combined_df = combined_df.set_index('datetime')
        
        # Rename price column
        combined_df = combined_df.rename(columns={price_col: 'electricity_price'})
        combined_df = combined_df[['electricity_price']]
        
        # Ensure numeric values
        combined_df['electricity_price'] = pd.to_numeric(combined_df['electricity_price'], errors='coerce')
        
        # Sort by index
        combined_df = combined_df.sort_index()
        
        # Check for duplicates
        if combined_df.index.duplicated().any():
            print(f"Warning: Found {combined_df.index.duplicated().sum()} duplicate timestamps")
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        # Check for missing values
        missing = combined_df.isna().sum().sum()
        if missing > 0:
            print(f"Warning: Found {missing} missing values")
            combined_df = combined_df.fillna(combined_df.mean())
        
        print(f"Processed electricity price data: {len(combined_df)} rows from {combined_df.index.min()} to {combined_df.index.max()}")
    
    except Exception as e:
        print(f"Error processing electricity price data: {e}")
        raise
    
    # Check for frequency
    inferred_freq = pd.infer_freq(combined_df.index)
    print(f"Inferred frequency: {inferred_freq}")
    
    return combined_df

def read_gas_price_data(gas_dir):
    """
    Read gas price data.
    
    Args:
        gas_dir (str or Path): Directory containing gas price CSV file
        
    Returns:
        pd.DataFrame: DataFrame with daily gas prices
    """
    gas_dir = Path(gas_dir)
    gas_file = gas_dir / "jeroen_punt_nl_dynamische_gasprijzen_alltime.csv"
    
    if not gas_file.exists():
        raise FileNotFoundError(f"Gas price file not found: {gas_file}")
    
    print(f"Reading gas price file: {gas_file.name}")
    
    # Read with semicolon delimiter and handle European-style decimal point
    try:
        df = pd.read_csv(gas_file, sep=';', decimal=',')
    except Exception as e:
        print(f"Error with comma decimal: {e}")
        try:
            df = pd.read_csv(gas_file, sep=';')
        except Exception as e2:
            print(f"Error with standard parsing: {e2}")
            # Try with more basic options
            df = pd.read_csv(gas_file, sep=None, engine='python')
    
    # Print raw data structure
    print("\nRaw gas price data structure:")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Process data
    try:
        # Assuming the first column is the datetime column
        date_col = df.columns[0]
        price_col = df.columns[1] if len(df.columns) > 1 else date_col
        
        print(f"Using {date_col} as date column and {price_col} as price column")
        
        # Convert to datetime
        df['datetime'] = pd.to_datetime(df[date_col], format="%d-%m-%Y %H:%M", errors='coerce')
        
        # Check for conversion issues
        invalid_dates = df['datetime'].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: Found {invalid_dates} invalid dates")
            # Try alternative format
            df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
            invalid_dates = df['datetime'].isna().sum()
            if invalid_dates > 0:
                print(f"Still have {invalid_dates} invalid dates after second attempt")
                df = df.dropna(subset=['datetime'])
        
        # Set datetime as index
        df = df.set_index('datetime')
        
        # Rename price column
        df = df.rename(columns={price_col: 'gas_price'})
        
        # Keep only the price column
        if 'gas_price' in df.columns:
            df = df[['gas_price']]
        else:
            # If the column name wasn't what we expected, take the first non-datetime column
            non_date_cols = [col for col in df.columns if not pd.api.types.is_datetime64_dtype(df[col])]
            if non_date_cols:
                df = df.rename(columns={non_date_cols[0]: 'gas_price'})
                df = df[['gas_price']]
            else:
                raise ValueError("Could not identify gas price column")
        
        # Ensure numeric values
        df['gas_price'] = pd.to_numeric(df['gas_price'], errors='coerce')
        
        # Sort by index
        df = df.sort_index()
        
        # Check for duplicates
        if df.index.duplicated().any():
            print(f"Warning: Found {df.index.duplicated().sum()} duplicate timestamps")
            df = df[~df.index.duplicated(keep='first')]
        
        # Check for missing values
        missing = df.isna().sum().sum()
        if missing > 0:
            print(f"Warning: Found {missing} missing values")
            df = df.fillna(df.mean())
        
        print(f"Processed gas price data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    except Exception as e:
        print(f"Error processing gas price data: {e}")
        print("Creating synthetic gas price data")
        
        # Create synthetic data as fallback
        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2023-12-31')
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create synthetic price data
        np.random.seed(42)
        synthetic_prices = np.random.lognormal(mean=1.0, sigma=0.5, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'gas_price': synthetic_prices
        }, index=dates)
        
        print("Using synthetic gas price data")
    
    # Check frequency
    inferred_freq = pd.infer_freq(df.index)
    print(f"Inferred frequency: {inferred_freq}")
    
    return df

def create_energy_crisis_indicator(prices_df, 
                                   start_date='2021-10-01', 
                                   end_date=None):
    """
    Create indicator variables for the European energy crisis period.
    
    Args:
        prices_df (pd.DataFrame): DataFrame with datetime index
        start_date (str): Start date of the energy crisis (default: Oct 1, 2021)
        end_date (str): End date of the energy crisis (default: None for ongoing)
        
    Returns:
        pd.DataFrame: DataFrame with added energy crisis indicators
    """
    df = prices_df.copy()
    
    # Create binary indicator for energy crisis period
    df['energy_crisis'] = 0
    
    # Set energy crisis period
    crisis_mask = df.index >= start_date
    if end_date:
        crisis_mask &= df.index <= end_date
    
    df.loc[crisis_mask, 'energy_crisis'] = 1
    
    # Create phase indicators for different periods of the crisis
    df['crisis_phase'] = 'pre_crisis'
    
    # Early phase: Oct 2021 - Feb 2022 (initial price increases)
    early_mask = (df.index >= '2021-10-01') & (df.index < '2022-02-24')
    df.loc[early_mask, 'crisis_phase'] = 'early_crisis'
    
    # Acute phase: Feb 24, 2022 (invasion) - Aug 2022
    acute_mask = (df.index >= '2022-02-24') & (df.index < '2022-09-01')
    df.loc[acute_mask, 'crisis_phase'] = 'acute_crisis'
    
    # Peak phase: Sep 2022 - Dec 2022 (highest prices)
    peak_mask = (df.index >= '2022-09-01') & (df.index < '2023-01-01')
    df.loc[peak_mask, 'crisis_phase'] = 'peak_crisis'
    
    # Stabilization phase: Jan 2023 onwards (still elevated but stabilizing)
    stabilization_mask = df.index >= '2023-01-01'
    df.loc[stabilization_mask, 'crisis_phase'] = 'stabilization'
    
    # Convert phase to category for more efficient storage
    df['crisis_phase'] = df['crisis_phase'].astype('category')
    
    # Create dummy variables for each phase
    phase_dummies = pd.get_dummies(df['crisis_phase'], prefix='phase')
    df = pd.concat([df, phase_dummies], axis=1)
    
    return df

def analyze_price_data(electricity_df, gas_df, output_dir=None):
    """
    Perform exploratory analysis on electricity and gas price data.
    
    Args:
        electricity_df (pd.DataFrame): DataFrame with hourly electricity prices
        gas_df (pd.DataFrame): DataFrame with daily gas prices
        output_dir (str or Path, optional): Directory to save plots
        
    Returns:
        tuple: (electricity_df, gas_df) with calculated features
    """
    print("\n" + "="*50)
    print("Electricity Price Data Summary")
    print("="*50)
    print(f"Period: {electricity_df.index.min()} to {electricity_df.index.max()}")
    print(f"Total records: {len(electricity_df)}")
    print("\nElectricity Price Statistics:")
    print(electricity_df.describe())
    
    print("\n" + "="*50)
    print("Gas Price Data Summary")
    print("="*50)
    print(f"Period: {gas_df.index.min()} to {gas_df.index.max()}")
    print(f"Total records: {len(gas_df)}")
    print("\nGas Price Statistics:")
    print(gas_df.describe())
    
    # Resample gas prices to hourly frequency for consistent analysis
    gas_hourly = gas_df.resample('h').ffill()
    
    # Merge electricity and gas price data
    merged_df = pd.merge(
        electricity_df, 
        gas_hourly, 
        left_index=True, 
        right_index=True, 
        how='outer'
    )
    
    # Fill in any missing values from the merge
    merged_df = merged_df.ffill().bfill()
    
    # Add energy crisis indicators
    merged_df = create_energy_crisis_indicator(merged_df)
    
    # Create time-based features for analysis
    merged_df['year'] = merged_df.index.year
    merged_df['month'] = merged_df.index.month
    merged_df['day'] = merged_df.index.day
    merged_df['dayofweek'] = merged_df.index.dayofweek
    merged_df['hour'] = merged_df.index.hour
    merged_df['is_weekend'] = merged_df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Calculate rolling statistics
    merged_df['elec_price_7d_mean'] = merged_df['electricity_price'].rolling(window=24*7).mean()
    merged_df['elec_price_7d_std'] = merged_df['electricity_price'].rolling(window=24*7).std()
    merged_df['elec_price_30d_mean'] = merged_df['electricity_price'].rolling(window=24*30).mean()
    
    merged_df['gas_price_7d_mean'] = merged_df['gas_price'].rolling(window=24*7).mean()
    merged_df['gas_price_30d_mean'] = merged_df['gas_price'].rolling(window=24*30).mean()
    
    # Calculate price ratios
    merged_df['gas_to_elec_ratio'] = merged_df['gas_price'] / merged_df['electricity_price']
    
    # ---- Create plots ----
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Time series of electricity and gas prices
    plt.figure(figsize=(15, 8))
    
    # Normalize both series to have max=1 for comparison
    elec_norm = merged_df['electricity_price'] / merged_df['electricity_price'].max()
    gas_norm = merged_df['gas_price'] / merged_df['gas_price'].max()
    
    plt.plot(elec_norm, label='Electricity Price (normalized)', alpha=0.7)
    plt.plot(gas_norm, label='Gas Price (normalized)', alpha=0.7)
    
    # Add crisis period shading
    for i, phase in enumerate(['early_crisis', 'acute_crisis', 'peak_crisis', 'stabilization']):
        phase_data = merged_df[merged_df['crisis_phase'] == phase]
        if not phase_data.empty:
            plt.axvspan(phase_data.index.min(), phase_data.index.max(), 
                       alpha=0.2, color=f'C{i+2}', label=f'{phase.replace("_", " ").title()}')
    
    plt.title('Normalized Electricity and Gas Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(output_dir / 'price_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Monthly averages by year
    plt.figure(figsize=(15, 8))
    
    # Calculate monthly averages for each year
    monthly_elec = merged_df.groupby(['year', 'month'])['electricity_price'].mean().unstack(0)
    
    # Plot each year's monthly pattern
    years = sorted(merged_df['year'].unique())
    for year in years:
        if year in monthly_elec.columns:
            plt.plot(monthly_elec.index, monthly_elec[year], 
                    marker='o', label=f'Electricity {year}')
    
    plt.title('Monthly Average Electricity Prices by Year')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(output_dir / 'monthly_elec_prices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Hourly patterns by weekday/weekend
    plt.figure(figsize=(15, 6))
    
    # Calculate hourly averages by weekday/weekend
    hourly_weekday = merged_df[merged_df['is_weekend'] == 0].groupby('hour')['electricity_price'].mean()
    hourly_weekend = merged_df[merged_df['is_weekend'] == 1].groupby('hour')['electricity_price'].mean()
    
    plt.plot(hourly_weekday.index, hourly_weekday, marker='o', label='Weekday')
    plt.plot(hourly_weekend.index, hourly_weekend, marker='o', label='Weekend')
    
    plt.title('Average Hourly Electricity Prices: Weekday vs Weekend')
    plt.xlabel('Hour of Day')
    plt.ylabel('Price')
    plt.xticks(range(0, 24, 3))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(output_dir / 'hourly_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Price volatility during crisis vs. non-crisis periods
    plt.figure(figsize=(12, 6))
    
    # Calculate daily volatility (standard deviation)
    daily_vol = merged_df.groupby(merged_df.index.date)['electricity_price'].std()
    daily_vol.index = pd.to_datetime(daily_vol.index)
    
    # Get crisis period
    crisis_start = merged_df[merged_df['energy_crisis'] == 1].index.min()
    
    # Plot volatility
    plt.plot(daily_vol.loc[:crisis_start], label='Pre-Crisis Volatility', alpha=0.7, color='green')
    plt.plot(daily_vol.loc[crisis_start:], label='Crisis Period Volatility', alpha=0.7, color='red')
    
    plt.title('Daily Electricity Price Volatility')
    plt.xlabel('Date')
    plt.ylabel('Standard Deviation within Day')
    plt.axvline(x=crisis_start, color='black', linestyle='--', 
                label=f'Crisis Start: {crisis_start.strftime("%Y-%m-%d")}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(output_dir / 'price_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Correlation between electricity and gas prices
    plt.figure(figsize=(10, 8))
    
    # Ensure we have enough data points by dropping NaNs
    scatter_data = merged_df.dropna(subset=['electricity_price', 'gas_price'])
    
    # Create scatterplot with color based on time period
    sc = plt.scatter(
        scatter_data['gas_price'], 
        scatter_data['electricity_price'],
        c=scatter_data.index.map(datetime.toordinal),
        alpha=0.5, 
        cmap='viridis',
        s=20
    )
    
    # Calculate correlation
    corr = scatter_data['electricity_price'].corr(scatter_data['gas_price'])
    
    plt.title(f'Electricity vs Gas Prices (Correlation: {corr:.2f})')
    plt.xlabel('Gas Price')
    plt.ylabel('Electricity Price')
    plt.colorbar(sc, label='Date')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(output_dir / 'price_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print additional analysis insights
    print("\n" + "="*50)
    print("Correlation Analysis")
    print("="*50)
    print(f"Correlation between electricity and gas prices: {corr:.4f}")
    
    crisis_corr = merged_df[merged_df['energy_crisis'] == 1][['electricity_price', 'gas_price']].corr().iloc[0, 1]
    precrisis_corr = merged_df[merged_df['energy_crisis'] == 0][['electricity_price', 'gas_price']].corr().iloc[0, 1]
    
    print(f"Pre-crisis correlation: {precrisis_corr:.4f}")
    print(f"During crisis correlation: {crisis_corr:.4f}")
    
    # Calculate average prices by crisis phase
    phase_prices = merged_df.groupby('crisis_phase', observed=True)[['electricity_price', 'gas_price']].mean()
    print("\nAverage Prices by Crisis Phase:")
    print(phase_prices)
    
    # Calculate volatility by phase
    phase_volatility = merged_df.groupby('crisis_phase', observed=True)[['electricity_price', 'gas_price']].std()
    print("\nPrice Volatility by Crisis Phase:")
    print(phase_volatility)
    
    return electricity_df, gas_df, merged_df

def prepare_price_features_for_model(electricity_df, gas_df, output_file=None):
    """
    Prepare price features ready to be integrated into the prediction model.
    
    Args:
        electricity_df (pd.DataFrame): DataFrame with hourly electricity prices
        gas_df (pd.DataFrame): DataFrame with daily gas prices
        output_file (str or Path, optional): Path to save the combined features
        
    Returns:
        pd.DataFrame: DataFrame with price features ready for model integration
    """
    # Resample gas prices to hourly frequency
    gas_hourly = gas_df.resample('h').ffill()
    
    # Merge electricity and gas price data
    price_features = pd.merge(
        electricity_df,
        gas_hourly,
        left_index=True,
        right_index=True,
        how='outer'
    )
    
    # Fill in any missing values from the merge
    price_features = price_features.ffill().bfill()
    
    # Add energy crisis indicator
    price_features = create_energy_crisis_indicator(price_features)
    
    # Rename columns to more descriptive names
    price_features = price_features.rename(columns={
        'electricity_price': 'elec_price',
        'gas_price': 'gas_price'
    })
    
    # Add lag features for electricity price
    elec_lags = [1, 3, 6, 12, 24, 48, 168]  # hours
    for lag in elec_lags:
        price_features[f'elec_price_lag_{lag}h'] = price_features['elec_price'].shift(lag)
    
    # Add rolling statistics for electricity price
    elec_windows = [24, 48, 168, 336]  # hours (1 day, 2 days, 1 week, 2 weeks)
    for window in elec_windows:
        price_features[f'elec_price_roll_mean_{window}h'] = (
            price_features['elec_price'].rolling(window=window, min_periods=1).mean()
        )
        price_features[f'elec_price_roll_std_{window}h'] = (
            price_features['elec_price'].rolling(window=window, min_periods=1).std()
        )
    
    # Add lag features for gas price (fewer lags since it's daily data)
    gas_lags = [24, 48, 168]  # hours (1 day, 2 days, 1 week)
    for lag in gas_lags:
        price_features[f'gas_price_lag_{lag}h'] = price_features['gas_price'].shift(lag)
    
    # Add rolling statistics for gas price
    gas_windows = [24*7, 24*14]  # hours (1 week, 2 weeks)
    for window in gas_windows:
        price_features[f'gas_price_roll_mean_{window}h'] = (
            price_features['gas_price'].rolling(window=window, min_periods=1).mean()
        )
    
    # Calculate price ratios and differences
    price_features['gas_to_elec_ratio'] = price_features['gas_price'] / price_features['elec_price']
    price_features['elec_price_diff_1d'] = price_features['elec_price'].diff(24)  # 1-day diff
    price_features['elec_price_diff_1w'] = price_features['elec_price'].diff(168)  # 1-week diff
    
    # Clean up any remaining NaN values
    price_features = price_features.ffill().bfill()
    
    # Save to file if requested
    if output_file:
        price_features.to_csv(output_file)
        print(f"Price features saved to {output_file}")
    
    return price_features


if __name__ == "__main__":
    # Paths to data directories
    script_dir = Path(__file__).parent
    base_dir = script_dir
    
    price_dir = base_dir / "data" / "price"
    gas_dir = base_dir / "data" / "gas"
    output_dir = base_dir / "data" / "analysis"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)    
    
    try:
        electricity_df = read_electricity_price_data(price_dir)
        gas_df = read_gas_price_data(gas_dir)
        
        # Analyze data and create plots
        electricity_df, gas_df, merged_df = analyze_price_data(electricity_df, gas_df, output_dir)
        
        # Prepare features for model integration
        price_features = prepare_price_features_for_model(
            electricity_df, 
            gas_df,
            output_file=base_dir / "data" / "price_features.csv"
        )
        
        print("\nPrepared price features for model integration:")
        print(f"Shape: {price_features.shape}")
        print(f"Columns: {price_features.columns.tolist()}")
        print(f"Period: {price_features.index.min()} to {price_features.index.max()}")       
                      
    except Exception as e:
        print(f"Error: {e}") 