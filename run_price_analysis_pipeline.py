#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CO2 Emission Factor Price Analysis Pipeline

This script runs the complete pipeline for analyzing the relationship between
electricity/gas prices and CO2 emission factors in the Netherlands.

The pipeline includes:
1. Reading and processing price data
2. Creating energy crisis indicators
3. Analyzing price trends and patterns
4. Visualizing the relationship between prices and emissions
5. Integrating price features into the CO2 emission prediction model
6. Evaluating the impact of price features on prediction accuracy

Author: AI Assistant
Date: March 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Import our modules
try:
    import price_analysis
    from visualize_price_impact import main as visualize_main
except ImportError:
    print("Error: Required modules not found. Make sure price_analysis.py and visualize_price_impact.py exist.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the CO2 emission price analysis pipeline')
    
    parser.add_argument('--price-dir', type=str, default='data/price',
                        help='Directory containing electricity price data')
    
    parser.add_argument('--gas-dir', type=str, default='data/gas',
                        help='Directory containing gas price data')
    
    parser.add_argument('--output-dir', type=str, default='data/analysis',
                        help='Directory to save analysis outputs')
    
    parser.add_argument('--features-output', type=str, default='data/price_features.csv',
                        help='Path to save the prepared price features')
    
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip the visualization step')
    
    parser.add_argument('--skip-model-integration', action='store_true',
                        help='Skip the model integration step')
    
    parser.add_argument('--crisis-start', type=str, default='2022-02-24',
                        help='Start date of the energy crisis (default: Russia-Ukraine war start)')
    
    parser.add_argument('--crisis-end', type=str, default=None,
                        help='End date of the energy crisis (default: None, meaning ongoing)')
    
    return parser.parse_args()

def run_price_data_processing(args):
    """Run the price data processing step."""
    print("\n" + "="*80)
    print("STEP 1: Processing Price Data")
    print("="*80)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read electricity price data
    print("\nReading electricity price data...")
    try:
        electricity_df = price_analysis.read_electricity_price_data(args.price_dir)
        print(f"Successfully read electricity price data: {len(electricity_df)} records")
    except Exception as e:
        print(f"Error reading electricity price data: {e}")
        electricity_df = None
    
    # Read gas price data
    print("\nReading gas price data...")
    try:
        gas_df = price_analysis.read_gas_price_data(args.gas_dir)
        print(f"Successfully read gas price data: {len(gas_df)} records")
    except Exception as e:
        print(f"Error reading gas price data: {e}")
        gas_df = None
    
    # Create energy crisis indicator
    if electricity_df is not None:
        print("\nCreating energy crisis indicators...")
        crisis_start = pd.to_datetime(args.crisis_start)
        crisis_end = pd.to_datetime(args.crisis_end) if args.crisis_end else None
        
        electricity_df = price_analysis.create_energy_crisis_indicator(
            electricity_df, crisis_start, crisis_end
        )
        print("Energy crisis indicators created")
    
    # Analyze price data
    if electricity_df is not None:
        print("\nAnalyzing price data...")
        price_analysis.analyze_price_data(electricity_df, gas_df, args.output_dir)
        print("Price analysis completed")
    
    # Prepare price features for model
    if electricity_df is not None and gas_df is not None:
        print("\nPreparing price features for model integration...")
        price_analysis.prepare_price_features_for_model(
            electricity_df, gas_df, args.features_output
        )
        print(f"Price features saved to {args.features_output}")
    
    return electricity_df, gas_df

def run_visualization(args):
    """Run the visualization step."""
    print("\n" + "="*80)
    print("STEP 2: Visualizing Price-Emission Relationship")
    print("="*80)
    
    if args.skip_visualization:
        print("Visualization step skipped as requested")
        return
    
    # Check if price features file exists
    if not os.path.exists(args.features_output):
        print(f"Error: Price features file not found at {args.features_output}")
        print("Skipping visualization step")
        return
    
    print("\nGenerating visualizations...")
    try:
        # Call the main function from visualize_price_impact.py
        visualize_main()
        print("Visualization completed successfully")
    except Exception as e:
        print(f"Error during visualization: {e}")

def run_model_integration(args):
    """Run the model integration step."""
    print("\n" + "="*80)
    print("STEP 3: Integrating Price Features into Model")
    print("="*80)
    
    if args.skip_model_integration:
        print("Model integration step skipped as requested")
        return
    
    # Check if price features file exists
    if not os.path.exists(args.features_output):
        print(f"Error: Price features file not found at {args.features_output}")
        print("Skipping model integration step")
        return
    
    print("\nIntegrating price features into model...")
    print("Note: This step requires manual integration into train_model.py and evaluate_model.py")
    print("The following files have been updated to include price features:")
    print("  - train_model.py: Added code to load and incorporate price features")
    print("  - evaluate_model.py: Added price features to known_covariates")
    
    print("\nTo train the model with price features, run:")
    print("  python train_model.py")
    
    print("\nTo evaluate the model with price features, run:")
    print("  python evaluate_model.py")

def print_summary(electricity_df, gas_df, args):
    """Print a summary of the pipeline run."""
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    print(f"\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data summary
    if electricity_df is not None:
        try:
            min_date = electricity_df['datetime'].min() if 'datetime' in electricity_df.columns else "unknown"
            max_date = electricity_df['datetime'].max() if 'datetime' in electricity_df.columns else "unknown"
            print(f"\nElectricity price data: {len(electricity_df)} records from {min_date} to {max_date}")
        except Exception as e:
            print(f"\nElectricity price data: {len(electricity_df)} records (date range unavailable)")
    else:
        print("\nElectricity price data: Not available")
    
    if gas_df is not None:
        try:
            min_date = gas_df['datetime'].min() if 'datetime' in gas_df.columns else "unknown"
            max_date = gas_df['datetime'].max() if 'datetime' in gas_df.columns else "unknown"
            print(f"Gas price data: {len(gas_df)} records from {min_date} to {max_date}")
        except Exception as e:
            print(f"Gas price data: {len(gas_df)} records (date range unavailable)")
    else:
        print("Gas price data: Not available")
    
    # Files generated
    print("\nFiles generated:")
    print(f"  - Price features: {args.features_output}")
    print(f"  - Analysis outputs: {args.output_dir}/")
    
    # List visualization files
    if not args.skip_visualization and os.path.exists(args.output_dir):
        viz_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
        if viz_files:
            print("\nVisualizations generated:")
            for f in viz_files:
                print(f"  - {args.output_dir}/{f}")
    
    print("\nNext steps:")
    print("  1. Review the generated visualizations in the analysis directory")
    print("  2. Train the model with price features using train_model.py")
    print("  3. Evaluate the model with price features using evaluate_model.py")
    print("  4. Compare model performance with and without price features")

def main():
    """Main function to run the pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print pipeline header
    print("\n" + "="*80)
    print("CO2 EMISSION FACTOR PRICE ANALYSIS PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Price directory: {args.price_dir}")
    print(f"Gas directory: {args.gas_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Features output: {args.features_output}")
    print(f"Energy crisis start: {args.crisis_start}")
    print(f"Energy crisis end: {args.crisis_end if args.crisis_end else 'Ongoing'}")
    
    # Run the pipeline steps
    electricity_df, gas_df = run_price_data_processing(args)
    run_visualization(args)
    run_model_integration(args)
    
    # Print summary
    print_summary(electricity_df, gas_df, args)

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    try:
        main()
    except Exception as e:
        print(f"\nError: Pipeline failed with error: {e}")
        sys.exit(1) 