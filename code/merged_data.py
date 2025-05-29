#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script processes ocean current data (ITF, AMOC, and Agulhas) from raw CSV files.
It merges the datasets based on common time points and saves them in decimal year format.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: May 28, 2025
"""

import os
import pandas as pd
from datetime import datetime

# Main processing function
def process_ocean_data():
    """
    Main function that processes ocean current data from raw CSV files.
    Performs the following steps:
    1. Sets input/output directory paths
    2. Reads raw data from CSV files
    3. Converts time to decimal year format
    4. Merges datasets in various combinations
    5. Saves processed data to output directory
    """
    # Define directory paths
    raw_data_dir = '../data/raw_data'
    processed_data_dir = '../data/processed_data'
    os.makedirs(processed_data_dir, exist_ok=True)  # Create output directory if needed

    # Load raw datasets with date parsing
    print("Loading raw datasets...")
    itf_df = pd.read_csv(
        os.path.join(raw_data_dir, 'itf_ts.csv'),
        parse_dates=['time']
    )
    amoc_df = pd.read_csv(
        os.path.join(raw_data_dir, 'amoc_ts.csv'),
        parse_dates=['time']
    )
    agulhas_df = pd.read_csv(
        os.path.join(raw_data_dir, 'aghulas_ts.csv'),
        parse_dates=['time']
    )
    print("Raw datasets loaded successfully.")

    # Convert time to decimal year for all datasets
    print("Converting time to decimal year format...")
    for df in [itf_df, amoc_df, agulhas_df]:
        df['time'] = df['time'].apply(convert_to_decimal_year)

    # Merge datasets in various combinations
    print("Merging datasets...")
    
    # 1. ITF and AMOC
    itf_amoc = pd.merge(
        itf_df, 
        amoc_df, 
        on='time',
        how='inner'
    )
    
    # 2. ITF and Agulhas
    itf_agulhas = pd.merge(
        itf_df, 
        agulhas_df, 
        on='time',
        how='inner'
    )
    
    # 3. Agulhas and AMOC
    agulhas_amoc = pd.merge(
        agulhas_df, 
        amoc_df, 
        on='time',
        how='inner'
    )
    
    # 4. All three datasets
    itf_agulhas_amoc = pd.merge(
        itf_df, 
        agulhas_df, 
        on='time',
        how='inner'
    ).merge(
        amoc_df, 
        on='time',
        how='inner'
    )

    # Save processed datasets
    print("Saving processed data...")
    itf_amoc.to_csv(
        os.path.join(processed_data_dir, 'itf_amoc.csv'), 
        index=False
    )
    itf_agulhas.to_csv(
        os.path.join(processed_data_dir, 'itf_agulhas.csv'), 
        index=False
    )
    agulhas_amoc.to_csv(
        os.path.join(processed_data_dir, 'agulhas_amoc.csv'), 
        index=False
    )
    itf_agulhas_amoc.to_csv(
        os.path.join(processed_data_dir, 'itf_agulhas_amoc.csv'), 
        index=False
    )
    print(f"Processing complete. Files saved to {processed_data_dir}")

def convert_to_decimal_year(date):
    """
    Convert datetime object to decimal year format.
    
    Args:
        date: datetime object to convert
        
    Returns:
        float: Decimal representation of the year
    """
    year_start = datetime(date.year, 1, 1)
    year_end = datetime(date.year + 1, 1, 1)
    elapsed = (date - year_start).total_seconds()
    total_seconds = (year_end - year_start).total_seconds()
    return date.year + elapsed / total_seconds

if __name__ == "__main__":
    print("Starting ocean data processing...")
    process_ocean_data()
    print("Script execution completed.")
