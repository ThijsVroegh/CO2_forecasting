
"""
Validation script for the Workalendar implementation.

This script generates holiday and vacation data for 2023-2024 using 
the Workalendar implementation, then saves the results to CSV files for reference.
"""
from pathlib import Path
import sys
import os
import datetime
import pandas as pd
from workalendar.europe import Netherlands, NetherlandsWithSchoolHolidays

# Add parent directory to path to import train_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import (
    get_dutch_holidays,
    get_dutch_school_vacations,
    add_holiday_features,
    add_vacation_features
)

def validate_holidays(years=(2023, 2024)):
    """Generate and save Dutch holidays for validation."""
    all_holidays = []
    
    # Get holidays for each year
    for year in years:
        holidays = get_dutch_holidays(year)
        for date, name in holidays.items():
            all_holidays.append({
                'date': date,
                'name': name,
                'year': year
            })
    
    # Create DataFrame and save to CSV
    holidays_df = pd.DataFrame(all_holidays)
    holidays_df = holidays_df.sort_values('date')
    
    output_path = Path('validation_holidays.csv')
    holidays_df.to_csv(output_path, index=False)
    print(f"Saved holiday validation data to {output_path}")
    
    return holidays_df

def validate_vacations(years=(2023, 2024)):
    """Generate and save Dutch school vacations for validation."""
    all_vacations = []
    
    # Get vacations for each year
    for year in years:
        vacations = get_dutch_school_vacations(year)
        for vacation_type, regions in vacations.items():
            for region, period in regions.items():
                all_vacations.append({
                    'start_date': period['start'],
                    'end_date': period['end'],
                    'type': vacation_type,
                    'region': region,
                    'year': year
                })
    
    # Create DataFrame and save to CSV
    vacations_df = pd.DataFrame(all_vacations)
    vacations_df = vacations_df.sort_values(['start_date', 'type', 'region'])
    
    output_path = Path('validation_vacations.csv')
    vacations_df.to_csv(output_path, index=False)
    print(f"Saved vacation validation data to {output_path}")
    
    return vacations_df

def validate_feature_computation():
    """Validate that the feature functions produce consistent output."""
    # Create a test dataframe with hourly data for 2023
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31 23:00:00')
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    test_df = pd.DataFrame(index=date_range)
    
    # Add holiday features
    holiday_df = add_holiday_features(test_df)
    holiday_features = holiday_df[['is_holiday', 'is_working_day', 'holiday_name']]
    
    # Add vacation features
    vacation_df = add_vacation_features(test_df)
    vacation_features = vacation_df[['is_school_vacation', 'is_summer_vacation', 'vacation_type']]
    
    # Combine features
    combined_df = pd.concat([holiday_features, vacation_features], axis=1)
    
    # Save to CSV
    output_path = Path('validation_features.csv')
    combined_df.to_csv(output_path)
    print(f"Saved feature validation data to {output_path}")
    
    # Print summary statistics
    print("\nFeature summary:")
    print(f"Total days in 2023: {len(date_range) // 24}")
    print(f"Holiday days: {(holiday_df['is_holiday'] == 1).sum() // 24}")
    print(f"Working days: {(holiday_df['is_working_day'] == 1).sum() // 24}")
    print(f"School vacation days: {(vacation_df['is_school_vacation'] == 1).sum() // 24}")
    print(f"Summer vacation days: {(vacation_df['is_summer_vacation'] == 1).sum() // 24}")
    
    return combined_df

def check_workalendar_capabilities():
    """Check what calendar capabilities are available in the Workalendar library."""
    # Basic Netherlands calendar
    cal = Netherlands()
    
    # Check working days
    today = datetime.date.today()
    is_working_today = cal.is_working_day(today)
    
    # Get holidays for current year
    current_year = datetime.datetime.now().year
    holidays = dict(cal.holidays(current_year))
    
    print("\nWorkalendar Netherlands capabilities:")
    print(f"Today ({today}) is a working day: {is_working_today}")
    
    print(f"\nHolidays for {current_year}:")
    for date, name in sorted(holidays.items()):
        print(f"  {date.strftime('%Y-%m-%d')}: {name}")
    
    # Check NetherlandsWithSchoolHolidays capabilities
    print("\nTesting NetherlandsWithSchoolHolidays:")
    
    regions = ["north", "middle", "south"]
    for region in regions:
        cal_with_school = NetherlandsWithSchoolHolidays(region=region)
        school_days = cal_with_school.get_variable_days(current_year)
        
        # Filter for school holidays only
        school_holidays = [day for day in school_days if "holiday" in day[1].lower()]
        
        print(f"\nRegion '{region}' has {len(school_holidays)} school holiday days in {current_year}")
        
        # Show sample of school holidays (if any)
        vacation_types = set([holiday[1] for holiday in school_holidays])
        print(f"Vacation types: {', '.join(sorted(vacation_types))}")

if __name__ == "__main__":
    print("Validating Workalendar implementation...")
    
    # Check Workalendar capabilities
    check_workalendar_capabilities()
    
    # Validate holidays
    holidays_df = validate_holidays()
    
    # Validate vacations
    vacations_df = validate_vacations()
    
    # Validate feature computation
    features_df = validate_feature_computation()
    
    print("\nValidation complete. Please check the CSV files for results.") 