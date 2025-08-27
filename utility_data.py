import pandas as pd
import numpy as np


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
        # Read numeric facts data (financial metrics with tags)
        numericFacts = pd.read_csv( f'{crt_dir}/num.txt',  sep='\t', low_memory=False )
        # Join numeric facts with submission data on 'adsh' (accession number)
        joined = numericFacts.join(submissions.set_index('adsh'), on='adsh' )
        
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
