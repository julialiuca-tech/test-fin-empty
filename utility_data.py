import pandas as pd
import numpy as np
import requests

DATA_DIR = 'data/'

# Quarter boundaries mapping for time interval calculations
QUARTER_DAYS = {0: 0, 
                1: 91, 
                2: 182, 
                3: 273, 
                4: 365, 
                5: 456, 
                6: 547, 
                7: 638, 
                8: 730 }

# Reverse mapping for efficient lookup: days -> quarter_number
DAYS_TO_QUARTER = {v: k for k, v in QUARTER_DAYS.items()}


def clean_cik_dup_submissions(submissions):
    """
    Removes companies that filed multiple submissions for the same (form, period) combination.
    
    A company (cik) should file only one submission (adsh) for any given (form, period) tuple.
    If multiple submissions are found for the same (cik, form, period), ALL submissions 
    from that combination are removed to maintain data quality.
    For instance, in 2022q4, we have 2 submissions submissions from
     (cik==1590364) & (period==20220930)]
    This breaks the segments feature. 
    
    Args:
        submissions (DataFrame): DataFrame containing submission data with columns 
                               ['adsh', 'cik', 'form', 'period', ...]
    Returns:
        DataFrame: Cleaned submissions dataframe with problematic combinations removed
    """
    initial_submission_count = len(submissions)
    
    # Identify companies with multiple submissions for the same (form, period)
    submission_counts = submissions.groupby(['cik', 'form', 'period']).size()
    multiple_submissions = submission_counts[submission_counts > 1]
    
    if len(multiple_submissions) > 0:
        print(f"  Found {len(multiple_submissions)} cases of multiple submissions for same (cik, form, period)")
        
        # Get the (cik, form, period) combinations that have multiple submissions
        problematic_combinations = multiple_submissions.index.tolist()
        
        # Create a mask to exclude these problematic submissions
        mask = True
        for cik, form, period in problematic_combinations:
            current_mask = ~((submissions['cik'] == cik) & 
                           (submissions['form'] == form) & 
                           (submissions['period'] == period))
            mask = mask & current_mask
        
        # Apply the mask to filter out problematic submissions
        submissions_cleaned = submissions[mask].copy()
        
        removed_count = initial_submission_count - len(submissions_cleaned)
        print(f"  Removed {removed_count} submissions from {len(problematic_combinations)} problematic (cik, form, period) combinations")
        
        return submissions_cleaned
    else:
        # No problematic submissions found, return original dataframe
        return submissions.copy()


def prep_data(data_dir):
    """
    Prepares the data for exploration. 
    Returns a dataframe with the joined data. 
    """ 
    df_joined = pd.DataFrame()
    
    for crt_dir in data_dir:
        print(f"Processing {crt_dir}...")
        # Read submission data (company information)
        submissions = pd.read_csv(f'{crt_dir}/sub.txt', sep='\t', low_memory=False, keep_default_na=False )
        submissions = submissions[
            ['adsh', 'cik', 'name', 'sic', 'fye', 'fy', 'fp', 'form', 'period']
        ]
        
        # Clean duplicate submissions for same (cik, form, period) combinations
        submissions = clean_cik_dup_submissions(submissions)
        
        # Read numeric facts data (financial metrics with tags)
        numericFacts = pd.read_csv( f'{crt_dir}/num.txt',  sep='\t', low_memory=False )
        # Join numeric facts with submission data on 'adsh' (accession number) - inner join
        joined = numericFacts.join(submissions.set_index('adsh'), on='adsh', how='inner')
        
        # Append to the main dataframe
        df_joined = pd.concat([df_joined, joined], ignore_index=True)
    
    df_joined['custom_tag'] = 1.0 * (df_joined['version'] == df_joined['adsh'])

    print(f"Total records loaded: {len(df_joined):,}")
    print(f"Total unique tags: {df_joined['tag'].nunique():,}")
    print(f"Total unique companies (CIK): {df_joined['cik'].nunique():,}")
    
    return df_joined


def top_tags(df_joined):
    """
    Utility function to explore SEC filing data: 
    1. Group by tag and count occurrences and distinct companies (cik)
    2. Display top 100 tags by occurrence count
    """
    # Group by tag and calculate statistics
    tag_stats = df_joined.groupby('tag').agg({
        'adsh': 'count',  # Count of occurrences
        'cik': 'nunique'  # Count of distinct companies
    }).rename(columns={
        'adsh': 'occurrences',
        'cik': 'distinct_companies'
    })
    
    # Sort by occurrences in descending order
    # NOTE: we sort by num of companies because some companies have ~6K segments 
    tag_stats_sorted = tag_stats.sort_values('distinct_companies', ascending=False)    
    tag_stats_sorted.to_csv('tag_stats_sorted.csv', index=True, header=True)


def read_tags_to_featurize(K_top_tags=50):
    """
    Reads tag statistics from CSV and creates a dataframe with top K tags to be used for featurization.
    
    Args:
        K_top_tags (int, default=50): Number of top tags to select
    
    Returns:
        DataFrame: DataFrame with columns ['rank', 'tag'] specifying which tags to featurize
    """
    print(f"Creating tags to featurize dataframe for top {K_top_tags} tags...")
    
    # Read the top tags from tag_stats_sorted.csv
    tag_stats = pd.read_csv(DATA_DIR + 'tag_stats_sorted.csv', index_col=0)
    df_tags_to_featurize = pd.DataFrame({
        'rank': range(1, K_top_tags + 1),
        'tag': tag_stats.head(K_top_tags).index.tolist()
    })
    
    print(f"Selected {len(df_tags_to_featurize)} tags for featurization:")
    print(f"Top 5 tags: {df_tags_to_featurize['tag'].head().tolist()}")
    
    return df_tags_to_featurize


def print_featurization(df_featurized, cik, period, tag, qtrs):
    """
    Utility function to print all non-null features for a specific (cik, period, tag, qtrs) combination.
    
    Args:
        df_featurized (DataFrame): The featurized dataframe from organize_feature_dataframe()
        cik (int): Company identifier
        period (str/int): Period identifier
        tag (str): Tag name (e.g., 'Assets', 'Revenues')
        qtrs (int): Number of quarters
    
    Returns:
        None: Prints the features to console
    """
    
    # Filter for the specific (cik, period) combination
    row_data = df_featurized[
        (df_featurized['cik'] == cik) & 
        (df_featurized['period'] == period)
    ]
    
    if len(row_data) == 0:
        print(f"No data found for CIK: {cik}, Period: {period}")
        return
    
    # Get the first (and should be only) matching row
    row = row_data.iloc[0]
    
    # Create the tag_qtrs pattern to look for
    tag_qtrs_pattern = f"{tag}_{qtrs}qtrs"
    
    # Find all columns that match this tag_qtrs pattern
    matching_columns = [col for col in df_featurized.columns if col.startswith(tag_qtrs_pattern)]
    
    if not matching_columns:
        print(f"No features found for Tag: {tag}, Qtrs: {qtrs}")
        print(f"Looking for pattern: {tag_qtrs_pattern}")
        return
    
    # Print header
    print(f"\n{'='*60}")
    print(f"FEATURES FOR CIK: {cik}, PERIOD: {period}, TAG: {tag}, QTRS: {qtrs}")
    print(f"{'='*60}")
    
    # Print non-null features
    found_features = False
    for col in sorted(matching_columns):
        value = row[col]
        if pd.notna(value):  # Check if value is not NaN
            feature_type = col.replace(tag_qtrs_pattern + '_', '')
            print(f"{feature_type:15}: {value:12.6f}")
            found_features = True
    
    if not found_features:
        print(f"All features for {tag_qtrs_pattern} are null/NaN")
    
    print(f"{'='*60}")


def get_cik_to_ticker_mapping():
    """
    Get CIK to ticker symbol mapping from SEC's company tickers JSON file.
    
    Returns:
        dict: Mapping of CIK (str) to ticker symbol (str)
    """
    try:
        # SEC's official company tickers JSON file
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            'User-Agent': 'SEC Data Analysis (your-email@domain.com)',  # SEC requires user agent
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to CIK -> ticker mapping
        cik_to_ticker = {}
        for entry in data.values():
            cik = str(entry['cik_str']).zfill(10)  # Pad CIK to 10 digits
            ticker = entry['ticker']
            cik_to_ticker[cik] = ticker
            
        print(f"✅ Loaded {len(cik_to_ticker)} CIK->ticker mappings from SEC")
        return cik_to_ticker
        
    except Exception as e:
        print(f"❌ Error loading CIK->ticker mapping: {e}")
        return {}
