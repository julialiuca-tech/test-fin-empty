#!/usr/bin/env python3
"""
Baseline Machine Learning Model for SEC Data

This script serves as the starting point for building a baseline ML model
using the featurized SEC financial data.

Current functionality:
- Loads simplified featurized data
- Ready for ML model development

Future ML pipeline components:
- Feature engineering and selection
- Model training and validation
- Performance evaluation
- Hyperparameter tuning
"""

import pandas as pd

# Import utility functions
from utility_binary_classifier import split_train_val_by_column, baseline_binary_classifier

COMPLETENESS_THRESHOLD = 0.2

featurized_data_file = 'data/featurized_2022/featurized_simplified.csv'

trend_horizon_in_months = 1
stock_trend_data_file = f'data/stock_202001_to_202507/price_trends_{trend_horizon_in_months}month.csv'
Y_LABEL = 'trend_5per_up' # can be 'trend_up_or_down' or 'trend_5per_up'

SPLIT_STRATEGY = {'cik': 'random'} # can be 'cik', 'date', or 'random'


def prepare_data_for_model(split_strategy=SPLIT_STRATEGY):
    """
    Load and join featurized data with stock trends, then split into train/val sets.
    
    Args:
        split_strategy (dict): Dictionary with one key-value pair for splitting strategy
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_cols)
    """
    
    # Load simplified featurized data 
    df_features = pd.read_csv(featurized_data_file)
    print(f"Features loaded: {df_features.shape}")
    
    # Load ground truth -- stock price trends
    df_trends = pd.read_csv(stock_trend_data_file)
    print(f"Trends loaded: {df_trends.shape}")

    # Join features and trends on cik and year_month resolution
    # Convert period and month_end_date to year_month for proper joining
    # Period is in YYYYMMDD format, so parse it correctly
    df_features['year_month'] = pd.to_datetime(df_features['period'], format='%Y%m%d').dt.to_period('M')
    # Handle timezone-aware dates by converting to naive datetime first
    df_trends['year_month'] = pd.to_datetime(df_trends['month_end_date']).dt.tz_localize(None).dt.to_period('M')
    # Inner join on cik and year_month
    df = df_features.merge(df_trends, on=['cik', 'year_month'], how='inner')
    print(f"Joined data: {df.shape}")

    # # Perform correlation analysis
    # print(f"\n" + "="*60)
    # print("Feature Correlation Analysis...")
    # correlations = correlation_analysis(df, Y_LABEL)

    # Use the split_strategy parameter - expect a dictionary with one key
    try:    
        by_column = list(split_strategy.keys())[0]
        split_for_training = split_strategy[by_column]
        print(f"Splitting data by {by_column} using {split_for_training} strategy")
        train_mask, val_mask = split_train_val_by_column(df, 0.7, by_column, split_for_training)
    except Exception as e:
        print(f"❌ Error with split_strategy: {str(e)}")
        print("🔄 Falling back to random splitting...")
        train_mask, val_mask = split_train_val_by_column(df, 0.7, None, 'random')

    # Prepare features and target
    feature_cols = [f for f in df.columns if '_current' in f or '_change' in f]
    X = df[feature_cols].copy()
    y = df[Y_LABEL].copy()
    
    # Apply masks to get training and validation sets
    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    
    return X_train, X_val, y_train, y_val, feature_cols


def select_feature_cols(df, strategy='all'):
    """
    Select feature columns based on strategy (all, completeness, current, change).
    
    Args:
        df (pd.DataFrame): DataFrame containing features
        strategy (str): Selection strategy
        
    Returns:
        list: Selected feature column names
    """
    # Identify feature columns
    feature_cols = [col for col in df.columns if '_current' in col or '_change' in col]
    if len(feature_cols) == 0:
        print("❌ No feature columns found for feature selection.")
        return []
    
    if strategy == 'all':
        return feature_cols 

    if strategy == 'completeness': 
        threshold = COMPLETENESS_THRESHOLD
        completeness = df[feature_cols].notna().mean()
        filtered_features = completeness[completeness >= threshold].index.tolist()
        return filtered_features
    
    if strategy == 'current': 
        return [col for col in feature_cols if '_current' in col]
    
    if strategy == 'change':
        return [col for col in feature_cols if '_change' in col]


def apply_imputation(X_train, X_val, imputation_strategy='none'): 
    """
    Apply imputation strategy to handle missing values.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        imputation_strategy (str): Strategy ('none' or 'median')
        
    Returns:
        tuple: (X_train_imputed, X_val_imputed)
    """
    if imputation_strategy == 'median':
        # Simple median imputation (baseline approach)
        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())  # Use training median for validation
        return X_train_imputed, X_val_imputed

    else: 
        return X_train.copy(), X_val.copy()



def build_baseline_model(X_train, X_val, y_train, y_val, feature_cols):
    """
    Build and evaluate baseline models with different feature selection and imputation strategies.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (pd.Series): Training target labels
        y_val (pd.Series): Validation target labels
        feature_cols (list): List of feature column names
        
    Returns:
        dict: Results dictionary with performance metrics
    """
    print(f"\n" + "="*60)
    print("Testing Different Missing Value Handling Approaches...")

    results = {}
    feature_selection_strategy_list = ['completeness', 'current', 'change', 'all']
    imputation_strategy_list = [ 'none', 'median']
    model_type_list = ['rf', 'xgb']
    X_train_orig, X_val_orig = X_train.copy(), X_val.copy()

    for selection in feature_selection_strategy_list:
         print(f"\n🔍 Testing strategy: {selection}")
         feature_cols = select_feature_cols(X_train_orig, selection)
         print(f"   Selected {len(feature_cols)} features")
         
         if len(feature_cols) == 0:
             print(f"❌ No features found for strategy '{selection}'. Skipping...")
             continue
         
         X_train_filtered = X_train_orig[feature_cols]
         X_val_filtered = X_val_orig[feature_cols]
         print(f"   Training data shape: {X_train_filtered.shape}")
         print(f"   Validation data shape: {X_val_filtered.shape}")
         
         for imputation in imputation_strategy_list:
             X_train_imputed, X_val_imputed = apply_imputation(X_train_filtered, X_val_filtered, imputation)
             for model_name in model_type_list:
                 try:
                     result = baseline_binary_classifier(X_train_imputed, X_val_imputed, y_train, y_val, model_name)
                     results[f"{selection}_{imputation}_{model_name}"] = result
                 except Exception as e:
                     print(f"❌ Error with {selection}_{imputation}_{model_name}: {str(e)}")
                     results[f"{selection}_{imputation}_{model_name}"] = None

    
    # 4. Compare results and recommend best approach
    print(f"\n" + "="*80)
    print("Model Comparison Results:")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for approach_name, result in results.items():
        if result is not None:
            comparison_data.append({
                'Approach': approach_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'ROC-AUC': f"{result['roc_auc']:.4f}"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best approach by ROC-AUC
        best_approach = max([k for k, v in results.items() if v is not None], 
                          key=lambda k: results[k]['roc_auc'])
        best_result = results[best_approach]
        
        print(f"\n🏆 Best Approach: {best_approach}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   Precision: {best_result['precision']:.4f}")
        print(f"   Recall: {best_result['recall']:.4f}")
        print(f"   ROC-AUC: {best_result['roc_auc']:.4f}")
        
        # Show top 10 features
        print(f"\n🔍 Top 10 Most Important Features:")
        print("=" * 50)
        for i, row in best_result['feature_importance'].head(10).iterrows():
            print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    return results






def main():
    """
    Run the complete SEC data analysis and ML pipeline.
    
    Returns:
        dict: Results dictionary with performance metrics
    """
    print("🚀 Starting SEC Data Analysis and ML Pipeline")
    print("=" * 60)
    
    # Prepare data (you can change split_strategy to 'date' for time-based splitting)
    X_train, X_val, y_train, y_val, feature_cols = prepare_data_for_model(split_strategy=SPLIT_STRATEGY)
 
    # Build and compare models
    print(f"\n" + "="*60)
    print("Building and Comparing ML Models...")
    results = build_baseline_model(X_train, X_val, y_train, y_val, feature_cols)
    
    return results


if __name__ == "__main__":
    results = main()
