#!/usr/bin/env python3
"""
SEC Stock Data Download Module

This module provides functions to download stock price data for companies
identified by their SEC CIK (Central Index Key) numbers. It processes featurized
data, maps CIKs to ticker symbols, and downloads historical stock prices in batches.

Key Functions:
- collect_cik_ticker_pairs(): Maps CIKs to ticker symbols from featurized data
- download_stock_data(): Downloads stock data in batches with rate limiting
- download_missed_tickers(): Downloads only tickers that were missed in previous runs
- closing_price_single_ticker(): Fetches closing prices for individual tickers
- closing_price_batch(): Processes a batch of tickers and saves results

Dependencies:
- pandas, numpy, requests, yfinance
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from utility_data import get_cik_to_ticker_mapping
import yfinance as yf
from featurize import FEATURIZED_SIMPLIFIED_FILE

NUM_BATCHES = 10
MAX_RETRIES = 3
SLEEP_TIME = 20 
STOCK_DIR = '/Users/juanliu/Workspace/git_test/SEC_data_explore/stock_data/'

processed_data_dir='/Users/juanliu/Workspace/git_test/SEC_data_explore/processed_data/'
featurized_file='featurized_simplified.csv' 

start_date = '2020-01-01' 
end_date = '2025-07-01'


def closing_price_single_ticker(stock, start_date, end_date):
    """
    Fetch closing price data for a single ticker using Yahoo Finance.
    
    This function downloads historical closing prices for a given ticker symbol
    over a specified date range. It includes basic error handling and rate limiting.
    
    Args:
        stock (str): Ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with Date and Close columns if successful, 
                     empty DataFrame if failed
    """
    tickerSymbol = stock

    try:
        tickerData = yf.Ticker(stock)
        tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
        if not tickerDf.empty:
            close_df = tickerDf[['Close']].copy()
            close_df = close_df.reset_index()  # Move Date from index to column
            close_df.columns = ['Date', 'Close']  # Rename columns for clarity 
            return close_df
        else:
            return pd.DataFrame()
            
    except Exception as e: 
        print(f"  ⚠️ Error fetching {tickerSymbol}: {type(e).__name__}")
        time.sleep(SLEEP_TIME)
    
            
def closing_price_batch(batch_num, cik_ticker_list, start_date, end_date):
    """
    Process a single batch of (cik, ticker) pairs to fetch closing price data.
    
    This function processes a list of (cik, ticker) tuples, fetches stock data for each ticker,
    and saves the results to a CSV file. It includes sleep delays between ticker calls to
    avoid rate limiting and provides detailed progress reporting.
    
    Args:
        batch_num (int): Batch number for file naming (e.g., 1, 2, 3...)
        cik_ticker_list (list): List of (cik, ticker) tuples to process
        start_date (str): Start date for stock data in YYYY-MM-DD format
        end_date (str): End date for stock data in YYYY-MM-DD format
        
    Returns:
        tuple: (successful_data_list, failed_tickers_list, success_count, failed_count)
               - successful_data_list: List of DataFrames with successful stock data
               - failed_tickers_list: List of ticker symbols that failed
               - success_count: Number of successful tickers
               - failed_count: Number of failed tickers
    """
    batch_data = []
    failed_tickers = []
    successful_count = 0
    failed_count = 0
    
    for i, (cik, ticker) in enumerate(cik_ticker_list):
        # Debugging output: print every 10 tickers
        # if (i + 1) % 10 == 0:
        #     print(f"  📈 Processed {i + 1}/{len(cik_ticker_list)} tickers")
        
        # Fetch stock data for this ticker
        stock_df = closing_price_single_ticker(ticker, start_date, end_date) 
        time.sleep(SLEEP_TIME)
        
        # Show progress for each ticker
        print(f"  🔄 Processing {ticker} ({i + 1}/{len(cik_ticker_list)})..., {stock_df.shape}")

        if not stock_df.empty:
            # Add CIK and ticker columns
            stock_df = stock_df.copy()
            stock_df['cik'] = cik
            stock_df['ticker'] = ticker
            
            batch_data.append(stock_df)
            successful_count += 1
        else:
            failed_tickers.append(ticker)
            failed_count += 1
    
    # (1) Write batch_data to file
    if batch_data:
        batch_df = pd.concat(batch_data, ignore_index=True)
        batch_filename = STOCK_DIR + f"batch_{batch_num}.csv"
        batch_df.to_csv(batch_filename, index=False)
        print(f"💾 Batch data saved to: {batch_filename}")
    
    # (2) Print successful count and failed count
    print(f"📊 Batch results: {successful_count} successful, {failed_count} failed")
    
    # (3) Print failed tickers
    if failed_tickers:
        print(f"❌ Failed tickers: {', '.join(failed_tickers)}")
    else:
        print("✅ All tickers processed successfully")
    
    return batch_data, failed_tickers, successful_count, failed_count


def collect_cik_ticker_pairs(processed_data_dir, featurized_file):
    """
    Collect and map CIK to ticker symbol pairs from featurized data.
    
    This function reads the featurized data file, extracts distinct CIK values,
    maps them to ticker symbols using SEC data, and returns an ordered list
    of (cik, ticker) pairs sorted alphabetically by ticker symbol.
    
    Args:
        processed_data_dir (str): Directory containing the featurized data file
        featurized_file (str): Name of the featurized data file (e.g., 'featurized_simplified.csv')
        
    Returns:
        list: List of (cik, ticker) tuples sorted alphabetically by ticker symbol
        
    """
    # Step 1: Read distinct CIKs from featurized data
    featurized_path = os.path.join(processed_data_dir, featurized_file)
    if not os.path.exists(featurized_path):
        print(f"❌ Featurized data file not found: {featurized_path}") 
    
    print(f"📊 Reading featurized data from: {featurized_path}")
    df_featurized = pd.read_csv(featurized_path, low_memory=False)
    distinct_ciks = df_featurized['cik'].unique()
    print(f"📊 Found {len(distinct_ciks)} distinct CIK values")
    
    # Step 2: Get CIK to ticker mapping
    cik_to_ticker = get_cik_to_ticker_mapping()
    if not cik_to_ticker:
        print("❌ Failed to load CIK to ticker mapping") 
    
    # Step 3: Create ticker_list_ordered (ranked by alphabetical order of ticker)
    cik_ticker_pairs = []
    for cik in distinct_ciks:
        cik_str = str(cik).zfill(10)  # Pad CIK to 10 digits
        if cik_str in cik_to_ticker:
            ticker = cik_to_ticker[cik_str]
            cik_ticker_pairs.append((cik, ticker))
    
    # Sort by ticker name alphabetically
    ticker_list_ordered = sorted(cik_ticker_pairs, key=lambda x: x[1])
    print(f"📊 Created ordered ticker list with {len(ticker_list_ordered)} companies")
    
    return ticker_list_ordered

def download_stock_data(ticker_list_ordered, start_date, end_date):
    """
    Download stock data for all tickers in batches with rate limiting.
    
    This function takes a list of (cik, ticker) pairs and processes them in batches
    to download historical stock price data. It includes progress tracking and
    comprehensive error reporting for each batch.
    
    Args:
        ticker_list_ordered (list): List of (cik, ticker) tuples sorted by ticker symbol
        start_date (str): Start date for stock data in YYYY-MM-DD format
        end_date (str): End date for stock data in YYYY-MM-DD format
        
    Returns:
        None: Results are saved to individual batch CSV files in STOCK_DIR
        
    Note:
        This function processes tickers in batches defined by NUM_BATCHES constant.
        Each batch is saved as a separate CSV file named 'batch_{batch_num}.csv'.
    """ 
    batch_size = len(ticker_list_ordered) // NUM_BATCHES
    if len(ticker_list_ordered) % NUM_BATCHES != 0:
        batch_size += 1
    
    print(f"📊 Processing {len(ticker_list_ordered)} tickers in {NUM_BATCHES} batches (batch size: ~{batch_size})")
    
    all_stock_data = []
    total_processed = 0
    total_successful = 0
    total_failed = 0
    
    for batch_num in range(NUM_BATCHES):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(ticker_list_ordered))
        batch_tickers = ticker_list_ordered[start_idx:end_idx]
        
        if not batch_tickers:
            break
            
        print(f"\n🔄 Processing batch {batch_num + 1}/{NUM_BATCHES} ({len(batch_tickers)} tickers)")
        
        # Process this batch using the new function
        batch_data, failed_tickers, batch_successful, batch_failed = closing_price_batch(
            batch_num, batch_tickers, start_date, end_date
        )
        
        # Add batch results to totals 
        total_processed += len(batch_tickers)
        total_successful += batch_successful
        total_failed += batch_failed
        
        # Print batch results
        print(f"✅ Batch {batch_num + 1} completed: {batch_successful} successful, {batch_failed} failed")
        if failed_tickers:
            print(f"❌ Failed tickers in batch {batch_num + 1}: {', '.join(failed_tickers[:10])}")
            if len(failed_tickers) > 10:
                print(f"   ... and {len(failed_tickers) - 10} more")


def download_missed_tickers():
    """
    Download stock data for tickers that were missed in previous runs.
    
    This function identifies tickers that exist in the featurized data but are missing
    from the stock_data/ directory, then downloads them in a single batch.
    
    Process:
    1. Read all CSV files in stock_data/ directory to find existing (cik, ticker) pairs
    2. Get all expected (cik, ticker) pairs from featurized data
    3. Find missing tickers by comparing the two sets
    4. Download missing tickers using closing_price_batch()
    
    Returns:
        None: Missing ticker data is saved to batch_{NUM_BATCHES}.csv
    """
    print("🔍 Checking for missed tickers...")
    print("=" * 50)
    
    # Step 1: Read all existing stock data files to get (cik, ticker) pairs
    print("\n📊 Step 1: Reading existing stock data files...")
    existing_pairs = set()
    
    if not os.path.exists(STOCK_DIR):
        print(f"❌ Stock data directory not found: {STOCK_DIR}")
        return
    
    # Get all CSV files in the stock_data directory
    csv_files = [f for f in os.listdir(STOCK_DIR) if f.endswith('.csv')]
    print(f"📁 Found {len(csv_files)} CSV files in {STOCK_DIR}")
    
    for filename in csv_files:
        file_path = os.path.join(STOCK_DIR, filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            if 'cik' in df.columns and 'ticker' in df.columns:
                # Get unique (cik, ticker) pairs from this file
                file_pairs = set(zip(df['cik'], df['ticker']))
                existing_pairs.update(file_pairs)
                print(f"  📄 {filename}: {len(file_pairs)} unique pairs")
        except Exception as e:
            print(f"  ⚠️ Error reading {filename}: {e}")
    
    print(f"📊 Total existing (cik, ticker) pairs: {len(existing_pairs)}")
    
    # Step 2: Get all expected (cik, ticker) pairs from featurized data
    print("\n📊 Step 2: Getting expected ticker pairs from featurized data...")
    expected_pairs = set(collect_cik_ticker_pairs(processed_data_dir, featurized_file))
    print(f"📊 Total expected (cik, ticker) pairs: {len(expected_pairs)}")
    
    # Step 3: Find missing tickers
    print("\n📊 Step 3: Finding missing tickers...")
    missing_pairs = expected_pairs - existing_pairs
    print(f"📊 Missing (cik, ticker) pairs: {len(missing_pairs)}")
    
    if not missing_pairs:
        print("✅ No missing tickers found! All data is up to date.")
        return
    
    # Convert set back to list for processing
    missing_tickers_list = list(missing_pairs)
    print(f"📋 Missing tickers: {[ticker for _, ticker in missing_tickers_list[:10]]}")
    if len(missing_tickers_list) > 10:
        print(f"   ... and {len(missing_tickers_list) - 10} more")
    
    # Step 4: Download missing tickers
    print(f"\n📈 Step 4: Downloading {len(missing_tickers_list)} missing tickers...")
    print(f"🔄 Using batch number {NUM_BATCHES} for missing tickers")
    
    try:
        batch_data, failed_tickers, successful_count, failed_count = closing_price_batch(
            NUM_BATCHES, missing_tickers_list, start_date, end_date
        )
        
        print(f"\n✅ Missing tickers download completed!")
        print(f"📊 Results: {successful_count} successful, {failed_count} failed")
        
        if failed_tickers:
            print(f"❌ Still failed tickers: {', '.join(failed_tickers[:10])}")
            if len(failed_tickers) > 10:
                print(f"   ... and {len(failed_tickers) - 10} more")
        
    except Exception as e:
        print(f"❌ Error downloading missing tickers: {e}")


def main():
    """
    Main function to orchestrate the stock data download process.
    
    This function coordinates the entire workflow:
    1. Collects CIK-ticker pairs from featurized data
    2. Downloads stock data in batches with rate limiting
    3. Saves results to individual batch files
    
    Returns:
        None: Results are saved to CSV files in the STOCK_DIR directory
    """
    print("🚀 Starting SEC Stock Data Download Process")
    print("=" * 50)
    
    # Step 1: Collect CIK-ticker pairs
    print("\n📊 Step 1: Collecting CIK-ticker pairs...")
    ticker_list_ordered = collect_cik_ticker_pairs(processed_data_dir, featurized_file)
    
    if not ticker_list_ordered:
        print("❌ No ticker pairs found. Exiting.")
        return
    
    # Step 2: Download stock data in batches
    print(f"\n📈 Step 2: Downloading stock data for {len(ticker_list_ordered)} companies...")
    download_stock_data(ticker_list_ordered, start_date, end_date)
    
    print("\n✅ Stock data download process completed!")
    print(f"📁 Results saved to: {STOCK_DIR}")


def run_missed_tickers():
    """
    Convenience function to run only the missed tickers download.
    
    This is useful when you want to download only the tickers that were
    missed in previous runs, without re-downloading all data.
    
    Returns:
        None: Missing ticker data is saved to batch_{NUM_BATCHES}.csv
    """
    download_missed_tickers()


if __name__ == "__main__":
    # Uncomment the line below to run the full download process
    # main()
    
    # Uncomment the line below to run only the missed tickers download
    run_missed_tickers()