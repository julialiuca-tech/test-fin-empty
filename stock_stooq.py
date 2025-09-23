#!/usr/bin/env python3
"""
Stock Data Processing from Stooq Directories

This module processes stock data from Stooq directories, reading closing prices
from NASDAQ, NYSE, and NYSEMKT stock files and standardizing ticker symbols.

Functions:
- load_stooq_stock_data(): Main function to load and process all stock data
- process_stock_directory(): Helper function to process a specific stock directory
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
from utility_data import get_cik_ticker_mapping, price_trend
from utility_data import remove_cik_w_missing_month, filter_by_date_continuity, filter_by_date_range, filter_by_price_range

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory containing Stooq stock data
STOOQ_BASE_DIR = '/Users/juanliu/Workspace/git_test/stock_Stooq_daily_US/'
STOOQ_SAVE_DIR = '/Users/juanliu/Workspace/git_test/stock_Stooq_daily_US/derived_data/'


# Stock exchange directories and their corresponding output variable names
STOCK_EXCHANGES = {
    'nasdaq_stock*': 'df_nasdaq',
    'nyse_stock*': 'df_nyse', 
    'nysemkt_stock*': 'df_nysemkt'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def process_stock_directory(directory_pattern: str, base_dir: str = STOOQ_BASE_DIR) -> pd.DataFrame:
    """
    Process stock data from a specific directory pattern.
    
    Args:
        directory_pattern (str): Pattern to match directories (e.g., 'nasdaq_stock*')
        base_dir (str): Base directory path
        
    Returns:
        pd.DataFrame: Processed stock data with standardized ticker symbols
    """
    print(f"🔍 Processing {directory_pattern}...")
    
    # Find all matching directories
    search_pattern = os.path.join(base_dir, directory_pattern)
    directories = glob.glob(search_pattern)
    
    if not directories:
        print(f"⚠️  No directories found matching pattern: {search_pattern}")
        return pd.DataFrame()
    
    print(f"📁 Found {len(directories)} directories: {[os.path.basename(d) for d in directories]}")
    
    all_stock_data = []
    
    for directory in directories:
        print(f"  📂 Processing directory: {os.path.basename(directory)}")
        
        # Find all data files in the directory (both .csv and .txt)
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        txt_files = glob.glob(os.path.join(directory, "*.txt"))
        data_files = csv_files + txt_files
        
        if not data_files:
            print(f"    ⚠️  No data files found in {directory}")
            continue
        
        # Count preferred stocks (files with "_" or "-" in filename)
        preferred_stocks = [f for f in data_files if '_' in os.path.basename(f) or '-' in os.path.basename(f)]
        regular_stocks = [f for f in data_files if '_' not in os.path.basename(f) and '-' not in os.path.basename(f)]
        
        print(f"    📄 Found {len(data_files)} data files ({len(csv_files)} CSV, {len(txt_files)} TXT)")
        print(f"    🚫 Skipping {len(preferred_stocks)} preferred stocks (with '_' or '-')")
        print(f"    ✅ Processing {len(regular_stocks)} regular stocks")
        
        for data_file in data_files:
            try:
                # Skip preferred stocks (files with "_" or "-" in filename)
                filename = os.path.basename(data_file)
                if '_' in filename or '-' in filename:
                    continue
                
                # Check if file is empty
                if os.path.getsize(data_file) == 0:
                    continue
                
                # Read the data file (handle both CSV and TXT)
                if data_file.endswith('.csv'):
                    df = pd.read_csv(data_file)
                else:  # .txt file - Stooq format is comma-separated
                    df = pd.read_csv(data_file, sep=',')
                
                # Skip if dataframe is empty
                if df.empty:
                    continue
                
                # Check if the file has the expected columns (Stooq format uses <CLOSE>)
                if '<CLOSE>' not in df.columns and 'Close' not in df.columns:
                    continue
                
                # Extract ticker from filename (remove extension)
                ticker = os.path.splitext(os.path.basename(data_file))[0]
                
                # Remove '.US' suffix from ticker if present
                if ticker.endswith('.US'):
                    ticker = ticker[:-3]  # Remove last 3 characters (.US)
                
                # Handle different column name formats
                if '<CLOSE>' in df.columns:
                    # Stooq format
                    close_col = '<CLOSE>'
                    date_col = '<DATE>'
                else:
                    # Standard format
                    close_col = 'Close'
                    date_col = 'Date'
                
                # Create a new dataframe with ticker and closing prices
                stock_df = df[[date_col, close_col]].copy()
                stock_df['ticker'] = ticker
                stock_df['exchange'] = os.path.basename(directory)
                
                # Rename columns for consistency
                stock_df = stock_df.rename(columns={date_col: 'date', close_col: 'close_price'})
                
                # Reorder columns
                stock_df = stock_df[['ticker', 'exchange', 'date', 'close_price']]
                
                all_stock_data.append(stock_df)
                
            except Exception as e:
                print(f"    ❌ Error processing {os.path.basename(data_file)}: {str(e)}")
                continue
    
    if not all_stock_data:
        print(f"⚠️  No valid stock data found for {directory_pattern}")
        return pd.DataFrame()
    
    # Combine all stock data
    combined_df = pd.concat(all_stock_data, ignore_index=True)
    
    # Convert date column to datetime (handle YYYYMMDD format)
    combined_df['date'] = pd.to_datetime(combined_df['date'], format='%Y%m%d', errors='coerce')
    
    # Sort by ticker and date
    combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"✅ Processed {len(combined_df)} records for {len(combined_df['ticker'].unique())} unique tickers")
    
    return combined_df

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def load_stooq_stock_data() -> pd.DataFrame:
    """
    Load and process stock data from all Stooq directories.
    
    Returns:
        pd.DataFrame: Combined stock data with columns:
            - ticker: Stock ticker (uppercase, no .US suffix)
            - exchange: Exchange name (nasdaq_stocks_1, nyse_stocks_1, etc.)
            - date: Trading date
            - close_price: Closing price
            - cik: Central Index Key (mapped from ticker)
    """
    print("🚀 Starting Stooq stock data processing...")
    print(f"📁 Base directory: {STOOQ_BASE_DIR}")
    
    # Check if base directory exists
    if not os.path.exists(STOOQ_BASE_DIR):
        print(f"❌ Base directory does not exist: {STOOQ_BASE_DIR}")
        return pd.DataFrame()
    
    # Process each stock exchange and collect all data
    all_dataframes = []
    
    for directory_pattern, output_name in STOCK_EXCHANGES.items():
        print(f"\n{'='*60}")
        print(f"Processing {directory_pattern} -> {output_name}")
        print(f"{'='*60}")
        
        df = process_stock_directory(directory_pattern)
        if not df.empty:
            all_dataframes.append(df)
    
    # Combine all dataframes
    if not all_dataframes:
        print("❌ No data found in any exchange")
        return pd.DataFrame()
    
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Process ticker names: convert to uppercase and remove .US suffix
    print(f"\n🔧 Processing ticker names...")
    df_combined['ticker'] = df_combined['ticker'].str.upper()
    df_combined['ticker'] = df_combined['ticker'].str.replace('.US', '', regex=False)
    
    # Add CIK column using ticker mapping
    print(f"🔗 Adding CIK mappings...")
    cik_to_ticker, ticker_to_cik = get_cik_ticker_mapping() 
    
    df_combined['cik'] = df_combined['ticker'].map(ticker_to_cik)
    
    # Drop records with null CIK values
    initial_count = len(df_combined)
    df_combined = df_combined.dropna(subset=['cik'])
    dropped_count = initial_count - len(df_combined)
    print(f"🗑️  Dropped {dropped_count:,} records with null CIK values")
    
    # Report mapping statistics
    mapped_count = df_combined['cik'].notna().sum()
    total_count = len(df_combined)
    print(f"✅ Mapped {mapped_count:,} out of {total_count:,} records ({mapped_count/total_count*100:.1f}%)")
    

    # Reorder columns
    df_combined = df_combined[['ticker', 'cik', 'exchange', 'date', 'close_price']]
    
    # Sort by ticker and date
    df_combined = df_combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(df_combined):,}")
    print(f"Unique tickers: {len(df_combined['ticker'].unique()):,}")
    print(f"Unique exchanges: {len(df_combined['exchange'].unique()):,}")
    print(f"Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")
    
    return df_combined

def month_end_price_stooq(df_combined):
    """
    Extract month-end price from the combined stock data.
    """

    df_combined['year_month'] = df_combined['date'].dt.to_period('M')
    
    # Group by (cik, ticker, year_month) and find the record with the largest Date in each group
    # This gives us the last trading day of each month for each company
    month_end_df = df_combined.loc[df_combined.groupby(['cik', 'ticker', 'year_month'])['date'].idxmax()].copy()
    
    # Rename columns for clarity
    month_end_df = month_end_df.rename(columns={'date': 'month_end_date'})
    
    # Select only the columns we need
    month_end_df = month_end_df[['cik', 'ticker', 'month_end_date', 'close_price', 'year_month']].reset_index(drop=True)
    return month_end_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    month_end_price_file = STOOQ_SAVE_DIR + 'month_end_price_stooq.csv'

    if not os.path.exists(month_end_price_file):
        # Load all stock data
        df_combined = load_stooq_stock_data()
        
        # remove records with date outside of 2000-01-01 to 2025-07-01
        df_combined = filter_by_date_range(df_combined, 'date', start_date='2000-01-01', end_date='2025-07-01')

        # # remove records with large gaps in date
        # df_combined, removed_ticker_info = filter_by_date_continuity(df_combined, 'date', gap_in_days=7)
        # print('debugging trace: removed tickers with max gap > 7 days:', removed_ticker_info[:5])

        month_end_df = month_end_price_stooq(df_combined)
        month_end_df.to_csv(month_end_price_file, index=False)
    else:
        month_end_df = pd.read_csv(month_end_price_file)
        # Convert year_month back to Period objects (lost when saving to CSV)
        month_end_df['year_month'] = pd.to_datetime(month_end_df['year_month']).dt.to_period('M')

    month_end_df = remove_cik_w_missing_month(month_end_df)

    # filter by price range
    month_end_df = filter_by_price_range(month_end_df, 'close_price', min_price=1, max_price=1000)

    # Calculate trends
    for horizon in [3, 1]:
        trend_df = price_trend(month_end_df, trend_horizon_in_months=horizon)
        if len(trend_df) > 0: 
            # Save results to CSV
            output_file = os.path.join(STOOQ_SAVE_DIR, f'price_trends_{horizon}month.csv')
            trend_df.to_csv(output_file, index=False)
            print(f"\n💾 {horizon}-month trends saved to: {output_file}")




 
