import pandas as pd
from pathlib import Path

def process_knmi_data(file_path: str) -> pd.DataFrame:
    # Read the data with proper column names
    df = pd.read_csv(file_path, 
                     skiprows=1,  # Skip the header row
                     delimiter=',',
                     skipinitialspace=True,
                     names=['STN', 'YYYYMMDD', 'HH', 'T'])  # Specify column names
    
    # Print column names to debug
    print("Available columns:", df.columns.tolist())
    
    # Convert temperature from 0.1°C to °C
    df['T'] = df['T'].astype(float) / 10.0
    
    # Handle hour 24 by converting it to hour 0 of the next day
    next_day_mask = df['HH'] == 24
    df.loc[next_day_mask, 'HH'] = 0
    
    # Convert YYYYMMDD to datetime
    df['date'] = pd.to_datetime(df['YYYYMMDD'].astype(str), format='%Y%m%d')
    
    # Add one day where hour was 24
    df.loc[next_day_mask, 'date'] = df.loc[next_day_mask, 'date'] + pd.Timedelta(days=1)
    
    # Create final datetime by combining date and hour
    df['datetime'] = df['date'] + pd.to_timedelta(df['HH'], unit='h')
    
    # Select only datetime and temperature columns
    result_df = df[['datetime', 'T']].copy()
    
    # Rename temperature column to be more descriptive
    result_df = result_df.rename(columns={'T': 'temperature_celsius'})
    
    # Sort by datetime
    result_df = result_df.sort_values('datetime')
    
    return result_df

def main():
    # Define the file path
    file_path = Path('knmi_data/result.txt')
    
    try:
        # Process the data
        df = process_knmi_data(file_path)
        
        # Display the first few rows
        print("\nFirst few rows of processed data:")
        print(df.head())
        
        # Display some basic statistics
        print("\nBasic statistics of temperature:")
        print(df['temperature_celsius'].describe())
        
        # Save the processed data
        df.to_csv('knmi_data/processed_temperatures.csv', index=False)
        print("\nData saved to 'knmi_data/processed_temperatures.csv'")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()