"""
Visualize the relationship between electricity prices and CO2 emission factors
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import warnings
import matplotlib.dates as mdates
from scipy import stats
import re

# Set the style
plt.style.use('seaborn-v0_8')
sns.set_context("talk")

# Ensure the analysis directory exists
os.makedirs('data/analysis', exist_ok=True)

def read_price_features(file_path='data/price_features.csv'):
    """Read price features from CSV file."""
    
    print(f"Reading price data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, parse_dates=['datetime'])
        print(f"Read {len(df)} price data points from {df['datetime'].min()} to {df['datetime'].max()}")
        return df
    
    except Exception as e:
        print(f"Error reading price features: {e}")
        return None

def find_emission_files(base_dir='data'):
    """Find the most recent historical_combined_ emission data file in the
    downloaded folder."""
    
    # Path to downloaded folder containing historical_combined_ files
    download_path = os.path.join(base_dir, 'downloaded', 'historical_combined_*.csv')
    
    # Get all matching CSV files
    all_files = glob.glob(download_path)
    
    if not all_files:
        print(f"No historical_combined_*.csv files found in {os.path.join(base_dir, 'downloaded')}")
        return []
    
    # Extract date from filename for sorting
    def extract_date(filename):
        # Extract date pattern from filename after "historical_combined_"
        filename_base = os.path.basename(filename)
        # Check if filename follows the expected pattern
        if filename_base.startswith("historical_combined_"):
            # Get the part after "historical_combined_"
            date_part = filename_base.split("historical_combined_")[1]
            # Remove file extension if present
            date_part = date_part.split(".")[0]
            
            # Try to extract YYYYMMDD pattern
            date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', date_part)
            if date_match:
                return f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}"
        return ""  # Return empty string if pattern not found
    
    # Sort files by extracted date (most recent first)
    sorted_files = sorted(all_files, key=extract_date, reverse=True)
    
    if sorted_files:
        print(f"Found {len(sorted_files)} historical_combined_ files in downloaded folder")
        print(f"Most recent file: {os.path.basename(sorted_files[0])}")
        return [sorted_files[0]]  # Return only the most recent file
    
    return []

def read_emission_data(base_dir='data'):
    """Read emission data from the most recent CSV file in the historical folder."""
    emission_files = find_emission_files(base_dir)
    
    if emission_files:
        print(f"Reading emission data from {emission_files[0]}")
        try:
            df = pd.read_csv(emission_files[0])
            
            # Check if 'emissionfactor' column exists
            if 'emissionfactor' in df.columns:
                print(f"Using 'emissionfactor' column from the file")
                df = df.rename(columns={'emissionfactor': 'emission_factor'})
            else:
                print(f"Warning: 'emissionfactor' column not found in {emission_files[0]}")
                print(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Ensure datetime column
            date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'timestamp'])]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'datetime'})
            
            # Parse datetime if it's not already
            if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
            elif 'datetime' not in df.columns:
                print("Warning: No datetime column found. Using index as datetime.")
                df['datetime'] = pd.to_datetime(df.index)
            
            print(f"Read {len(df)} emission data points from {df['datetime'].min()} to {df['datetime'].max()}")
            return df
        except Exception as e:
            print(f"Error reading emission file: {e}")
    
    print("No emission data found in historical folder.")
    return None

def merge_data(price_df, emission_df):
    """Merge price and emission data on datetime."""
    
    if price_df is None or emission_df is None:
        print("Cannot merge data: missing price or emission data")
        return None
    
    # Ensure datetime is the index for both dataframes
    if 'datetime' in price_df.columns and 'datetime' in emission_df.columns:
        merged_df = pd.merge(price_df, emission_df, on='datetime', how='inner')
        print(f"Merged dataset contains {len(merged_df)} records from {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")
        return merged_df
    else:
        print("Cannot merge: datetime column missing in one or both dataframes")
        return None

def plot_price_vs_emissions(df, output_dir='data/analysis'):
    """Create scatter plot of electricity price vs emission factor."""
    
    if df is None or 'elec_price' not in df.columns or 'emission_factor' not in df.columns:
        print("Cannot create scatter plot: missing required columns")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 10))
    
    # Color points by energy crisis if available
    if 'energy_crisis' in df.columns:
        palette = {0: 'blue', 1: 'red'}
        hue = 'energy_crisis'
        hue_order = [0, 1]
        legend_labels = ['Pre-Crisis', 'During Crisis']
    
    else:
        # Use a continuous color scale based on date
        palette = 'viridis'
        hue = df['datetime'].astype(int)
        hue_order = None
        legend_labels = None
    
    # Create scatter plot
    scatter = sns.scatterplot(
        data=df, 
        x='elec_price', 
        y='emission_factor',
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        alpha=0.6,
        s=50
    )
    
    # Add regression line
    sns.regplot(
        data=df, 
        x='elec_price', 
        y='emission_factor',
        scatter=False,
        line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2}
    )
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(df['elec_price'], df['emission_factor'])
    
    plt.title(f'Electricity Price vs CO2 Emission Factor\nCorrelation: {corr:.3f} (p-value: {p_value:.3e})', fontsize=16)
    plt.xlabel('Electricity Price (€/MWh)', fontsize=14)
    plt.ylabel('CO2 Emission Factor (kg CO2/kWh)', fontsize=14)
    
    # Add custom legend if using energy crisis
    if 'energy_crisis' in df.columns:
        handles, _ = scatter.get_legend_handles_labels()
        plt.legend(handles, legend_labels, title='Energy Crisis Period', fontsize=12)
    
    # Add annotations
    plt.annotate(
        f'n = {len(df)}\nCorrelation: {corr:.3f}\np-value: {p_value:.3e}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=12
    )
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'price_vs_emissions.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved scatter plot to {output_file}")
    plt.close()

def plot_time_series(df, output_dir='data/analysis'):
    """Create time series plot of electricity price and emission factor."""
    
    if df is None or 'elec_price' not in df.columns or 'emission_factor' not in df.columns:
        print("Cannot create time series plot: missing required columns")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the dataframe with only numeric columns for resampling
    numeric_df = df[['datetime', 'elec_price', 'emission_factor']].copy()
    
    # If energy_crisis exists and is numeric, include it
    if 'energy_crisis' in df.columns:
        try:
            # Convert to numeric if it's not already
            energy_crisis = pd.to_numeric(df['energy_crisis'], errors='coerce')
            # Fill NaN values with 0
            energy_crisis = energy_crisis.fillna(0).astype(int)
            numeric_df['energy_crisis'] = energy_crisis
        except:
            print("Warning: Could not convert energy_crisis to numeric. Creating binary indicator.")
            # Create a binary indicator based on date (Ukraine war start)
            war_start = pd.Timestamp('2022-02-24')
            numeric_df['energy_crisis'] = (numeric_df['datetime'] >= war_start).astype(int)
    
    # Resample to daily for better visualization
    daily_df = numeric_df.set_index('datetime').resample('D').mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot electricity price
    color = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Electricity Price (€/MWh)', color=color, fontsize=14)
    ax1.plot(daily_df['datetime'], daily_df['elec_price'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for emission factor
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('CO2 Emission Factor (kg CO2/kWh)', color=color, fontsize=14)
    ax2.plot(daily_df['datetime'], daily_df['emission_factor'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add energy crisis shading if available
    if 'energy_crisis' in daily_df.columns:
        crisis_start = daily_df[daily_df['energy_crisis'] > 0.5]['datetime'].min()
        if not pd.isna(crisis_start):
            ax1.axvline(x=crisis_start, color='black', linestyle='--', linewidth=1.5)
            ax1.text(crisis_start, ax1.get_ylim()[1]*0.95, 'Energy Crisis Start', 
                    rotation=90, verticalalignment='top')
            
            # Shade the crisis period
            crisis_periods = daily_df[daily_df['energy_crisis'] > 0.5]
            if not crisis_periods.empty:
                for _, period in crisis_periods.groupby((crisis_periods['datetime'].diff() > pd.Timedelta(days=1)).cumsum()):
                    ax1.axvspan(period['datetime'].min(), period['datetime'].max(), 
                               alpha=0.2, color='gray')
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    
    plt.title('Electricity Price and CO2 Emission Factor Over Time', fontsize=16)
    fig.tight_layout()
    
    output_file = os.path.join(output_dir, 'price_emissions_time_series.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved time series plot to {output_file}")
    plt.close()

def plot_crisis_comparison(df, output_dir='data/analysis'):
    """Create boxplot comparing emission factors before and during energy crisis."""
    
    if df is None or 'emission_factor' not in df.columns:
        print("Cannot create crisis comparison plot: missing required columns")
        return
    
    if 'energy_crisis' not in df.columns:
        print("Cannot create crisis comparison plot: missing energy_crisis column")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    ax = sns.boxplot(
        data=df,
        x='energy_crisis',
        y='emission_factor',
        palette=['blue', 'red']
    )
    
    # Add swarmplot for data points (with small sample for visibility)
    sample_size = min(5000, len(df))
    sampled_df = df.sample(sample_size, random_state=42)
    sns.swarmplot(
        data=sampled_df,
        x='energy_crisis',
        y='emission_factor',
        color='black',
        alpha=0.5,
        size=3
    )
    
    # Perform t-test
    pre_crisis = df[df['energy_crisis'] == 0]['emission_factor']
    during_crisis = df[df['energy_crisis'] == 1]['emission_factor']
    t_stat, p_value = stats.ttest_ind(pre_crisis, during_crisis, equal_var=False)
    
    # Add statistics
    plt.title('CO2 Emission Factor: Before vs. During Energy Crisis', fontsize=16)
    plt.xlabel('')
    plt.ylabel('CO2 Emission Factor (kg CO2/kWh)', fontsize=14)
    
    # Set x-tick labels
    ax.set_xticklabels(['Pre-Crisis', 'During Crisis'])
    
    # Add annotations with statistics
    pre_mean = pre_crisis.mean()
    during_mean = during_crisis.mean()
    percent_change = ((during_mean - pre_mean) / pre_mean) * 100
    
    plt.annotate(
        f'Pre-Crisis Mean: {pre_mean:.4f}\n'
        f'During Crisis Mean: {during_mean:.4f}\n'
        f'Change: {percent_change:.1f}%\n'
        f't-statistic: {t_stat:.3f}\n'
        f'p-value: {p_value:.3e}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=12
    )
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'crisis_comparison.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved crisis comparison plot to {output_file}")
    plt.close()

def plot_monthly_patterns(df, output_dir='data/analysis'):
    """Create plot showing monthly patterns of prices and emissions."""
    
    if df is None or 'elec_price' not in df.columns or 'emission_factor' not in df.columns:
        print("Cannot create monthly patterns plot: missing required columns")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract month and calculate monthly averages
    month_df = df[['datetime', 'elec_price', 'emission_factor']].copy()
    month_df['month'] = month_df['datetime'].dt.month
    
    monthly_avg = month_df.groupby('month').agg({
        'elec_price': 'mean',
        'emission_factor': 'mean'
    }).reset_index()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot monthly electricity prices
    sns.barplot(data=monthly_avg, x='month', y='elec_price', ax=ax1, color='blue', alpha=0.7)
    ax1.set_title('Average Electricity Price by Month', fontsize=14)
    ax1.set_ylabel('Electricity Price (€/MWh)', fontsize=12)
    ax1.set_xlabel('')
    
    # Plot monthly emission factors
    sns.barplot(data=monthly_avg, x='month', y='emission_factor', ax=ax2, color='red', alpha=0.7)
    ax2.set_title('Average CO2 Emission Factor by Month', fontsize=14)
    ax2.set_ylabel('CO2 Emission Factor (kg CO2/kWh)', fontsize=12)
    ax2.set_xlabel('Month', fontsize=12)
    
    # Set month names on x-axis
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2.set_xticklabels(month_names)
    
    # Calculate correlation between monthly prices and emissions
    corr, p_value = stats.pearsonr(monthly_avg['elec_price'], monthly_avg['emission_factor'])
    
    # Add correlation annotation
    fig.suptitle(
        f'Monthly Patterns of Electricity Prices and CO2 Emissions\n'
        f'Monthly Correlation: {corr:.3f} (p-value: {p_value:.3e})',
        fontsize=16
    )
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'monthly_patterns.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved monthly patterns plot to {output_file}")
    plt.close()

def analyze_price_emission_relationship(df):
    """Analyze the relationship between prices and emissions."""
    
    if df is None or 'elec_price' not in df.columns or 'emission_factor' not in df.columns:
        print("Cannot analyze price-emission relationship: missing required columns")
        return
    
    # Calculate overall correlation
    corr, p_value = stats.pearsonr(df['elec_price'], df['emission_factor'])
    print(f"\nPrice-Emission Relationship Analysis:")
    print(f"Overall correlation: {corr:.4f} (p-value: {p_value:.3e})")
    
    # Analyze by price quartiles
    df_clean = df[['elec_price', 'emission_factor', 'datetime']].copy()
    df_clean['price_quartile'] = pd.qcut(df_clean['elec_price'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    quartile_stats = df_clean.groupby('price_quartile').agg({
        'elec_price': ['mean', 'min', 'max'],
        'emission_factor': ['mean', 'std', 'count']
    })
    
    print("\nEmission Factor by Price Quartile:")
    print(quartile_stats)
    
    # Analyze by time of day
    df_clean['hour'] = df_clean['datetime'].dt.hour
    
    # Group by hour and calculate correlation for each hour
    hourly_stats = []
    for hour, group in df_clean.groupby('hour'):
        if len(group) > 10:  # Ensure enough data points
            corr, p = stats.pearsonr(group['elec_price'], group['emission_factor'])
            hourly_stats.append({
                'hour': hour,
                'correlation': corr,
                'p_value': p,
                'count': len(group)
            })
    
    hourly_df = pd.DataFrame(hourly_stats)
    if not hourly_df.empty:
        print("\nCorrelation by Hour of Day (Top 3 positive and negative):")
        print("Strongest positive correlations:")
        print(hourly_df.sort_values('correlation', ascending=False).head(3)[['hour', 'correlation', 'p_value']])
        print("Strongest negative correlations:")
        print(hourly_df.sort_values('correlation').head(3)[['hour', 'correlation', 'p_value']])
    
    # Analyze by crisis period if available
    if 'energy_crisis' in df.columns:
        try:
            # Try to convert energy_crisis to numeric
            energy_crisis = pd.to_numeric(df['energy_crisis'], errors='coerce')
            df_clean['energy_crisis'] = energy_crisis.fillna(0).astype(int)
            
            crisis_stats = []
            for crisis, group in df_clean.groupby('energy_crisis'):
                if len(group) > 10:  # Ensure enough data points
                    corr, p = stats.pearsonr(group['elec_price'], group['emission_factor'])
                    crisis_stats.append({
                        'crisis': crisis,
                        'correlation': corr,
                        'p_value': p,
                        'count': len(group)
                    })
            
            crisis_df = pd.DataFrame(crisis_stats)
            if not crisis_df.empty:
                print("\nCorrelation by Crisis Period:")
                for _, row in crisis_df.iterrows():
                    period = "During Crisis" if row['crisis'] == 1 else "Pre-Crisis"
                    print(f"{period}: {row['correlation']:.4f} (p-value: {row['p_value']:.3e}, n={row['count']})")
        except Exception as e:
            print(f"Warning: Could not analyze by crisis period: {e}")

def main():
    """Main function to run the analysis."""
    
    # Read price data
    price_df = read_price_features()
    
    # Read emission data
    emission_df = read_emission_data()
    
    # Merge data
    merged_df = merge_data(price_df, emission_df)
    
    if merged_df is not None:
        # Create output directory
        output_dir = 'data/analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        plot_price_vs_emissions(merged_df, output_dir)
        plot_time_series(merged_df, output_dir)
        
        if 'energy_crisis' in merged_df.columns:
            plot_crisis_comparison(merged_df, output_dir)
        
        plot_monthly_patterns(merged_df, output_dir)
        
        # Analyze relationship
        analyze_price_emission_relationship(merged_df)
        
        print("\nVisualization process complete. Check the data/analysis directory for outputs.")
    else:
        print("Failed to create visualizations due to missing or incompatible data.")

if __name__ == "__main__":
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main() 