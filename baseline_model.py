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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

COMPLETENESS_THRESHOLD = 0.15

featurized_data_file = 'processed_data/featurized_simplified.csv'

trend_horizon_in_months = 1
stock_trend_data_file = f'stock_data/price_trends_{trend_horizon_in_months}month.csv'


def prepare_data_for_model():
    """
    Prepare and join featurized financial data with stock price trend data.
    
    This function loads the featurized financial data and stock price trend data,
    performs an inner join on CIK and year_month, and returns the prepared
    training and validation datasets.
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_cols)
            - X_train: Training features DataFrame
            - X_val: Validation features DataFrame  
            - y_train: Training target labels Series
            - y_val: Validation target labels Series
            - feature_cols: List of feature column names
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
    # correlations = correlation_analysis(df)

    # Prepare features and target
    feature_cols = [f for f in df.columns if '_current' in f or '_change' in f]
    X = df[feature_cols].copy()
    y = df['trend_up_or_down'].copy()
    
    # 2. Split samples by CIK into training and validation datasets
    print(f"\n🔄 Splitting data by CIK...")
    unique_ciks = df['cik'].unique()
    print(f"📊 Total unique companies (CIKs): {len(unique_ciks)}")
    
    # Split CIKs into train/validation sets
    train_ciks, val_ciks = train_test_split(unique_ciks, test_size=0.3, random_state=42)
    
    # Create train/validation masks based on CIK
    train_mask = df['cik'].isin(train_ciks)
    val_mask = df['cik'].isin(val_ciks)
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    
    print(f"📊 Training set: {len(X_train)} samples from {len(train_ciks)} companies")
    print(f"📊 Validation set: {len(X_val)} samples from {len(val_ciks)} companies")
    return X_train, X_val, y_train, y_val, feature_cols


def build_baseline_model(X_train, X_val, y_train, y_val, feature_cols):
    """
    Build and evaluate baseline models for stock trend prediction with different missing value handling approaches.
    
    This function tests multiple machine learning approaches including Random Forest and XGBoost
    with different missing value handling strategies. It compares performance across all approaches
    and identifies the best performing model.
    
    Args:
        X_train (pd.DataFrame): Training features with columns for financial metrics
        X_val (pd.DataFrame): Validation features with columns for financial metrics
        y_train (pd.Series): Training target labels (trend_up_or_down)
        y_val (pd.Series): Validation target labels (trend_up_or_down)
        feature_cols (list): List of feature column names
        
    Returns:
        dict: Dictionary containing results for each approach with keys:
            - 'accuracy': Model accuracy score
            - 'precision': Model precision score
            - 'recall': Model recall score
            - 'roc_auc': Model ROC-AUC score
            - 'model': Trained model object
            - 'feature_importance': DataFrame with feature importance rankings
    """
    # 1. Identify feature columns (those with '_current' or '_change' in name)

    
    # 3. Test different missing value handling approaches
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
                     result = test_model(X_train_imputed, X_val_imputed, y_train, y_val, model_name)
                     results[f"{selection}_{imputation}_{model_name}"] = result
                 except Exception as e:
                     print(f"❌ Error with {selection}_{imputation}_{model_name}: {str(e)}")
                     results[f"{selection}_{imputation}_{model_name}"] = None

    
    # 4. Compare results and recommend best approach
    print(f"\n" + "="*80)
    print("Model Comparison Results:")
    print(f"{'Approach':<40} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'Status'}")
    print("-" * 90)
    
    best_accuracy = 0
    best_approach = None
    
    for approach_name, result in results.items():
        if result is not None:
            status = "✅"
            accuracy = f"{result['accuracy']:.4f}"
            precision = f"{result['precision']:.4f}"
            recall = f"{result['recall']:.4f}"
            roc_auc = f"{result['roc_auc']:.4f}"
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_approach = approach_name
        else:
            status = "❌"
            accuracy = "N/A"
            precision = "N/A"
            recall = "N/A"
            roc_auc = "N/A"
        
        print(f"{approach_name:<40} {accuracy:<10} {precision:<10} {recall:<10} {roc_auc:<10} {status}")
    
    if best_approach:
        print(f"\n🏆 Best performing approach: {best_approach}")
        best_result = results[best_approach]
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   Precision: {best_result['precision']:.4f}")
        print(f"   Recall: {best_result['recall']:.4f}")
        print(f"   ROC-AUC: {best_result['roc_auc']:.4f}")
        
        # Show feature importance for best model
        if 'feature_importance' in best_result:
            print(f"\n🔍 Top 10 Most Important Features ({best_approach}):")
            feature_importance = best_result['feature_importance']
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    return results


def test_model(X_train, X_val, y_train, y_val, model_name): 
    feature_cols = X_train.columns

    if model_name == 'rf':
        # Random Forest can handle missing values natively by using NaN as a separate category
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, 
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    elif model_name == 'xgb':
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
    else:
        raise ValueError(f"Invalid model type: {model_name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model_name': model_name,
        'trained_model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc, 
        'feature_importance': feature_importance
    }


def apply_imputation(X_train, X_val, imputation_strategy='none'): 
    if imputation_strategy == 'median':
        # Simple median imputation (baseline approach)
        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())  # Use training median for validation
        return X_train_imputed, X_val_imputed

    else: 
        return X_train.copy(), X_val.copy()


def select_feature_cols(df, strategy='all'):
    """
    Select features 
        
    Returns:
        list: List of feature column names that meet the completeness threshold
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


def correlation_analysis(df):
    """
    Perform correlation analysis between features and trend_up_or_down label.
    
    This function analyzes the linear correlation between financial features and
    stock price trends. It identifies the most predictive features and provides
    insights into which financial metrics are most strongly associated with
    stock price movements.
    
    Args:
        df (pd.DataFrame): Joined dataset with features and target labels
        
    Returns:
        pd.Series: Sorted correlation coefficients (absolute values) for all features
    """
    # Identify feature columns (those with '_current' or '_change' in name)
    feature_cols = [col for col in df.columns if '_current' in col or '_change' in col]
    
    if len(feature_cols) == 0:
        print("❌ No feature columns found for correlation analysis.")
        return
    
    print(f"📊 Analyzing correlation for {len(feature_cols)} features...")
    
    # Prepare features and target
    X = df[feature_cols].copy()
    y = df['trend_up_or_down'].copy()
    
    # Handle missing values for correlation analysis
    X_clean = X.fillna(X.median())
    
    # Calculate correlations
    correlations = X_clean.corrwith(y).abs().sort_values(ascending=False)
    
    print(f"\n🔍 Top 20 Features by Absolute Correlation with trend_up_or_down:")
    print(f"{'Rank':<5} {'Feature':<40} {'Correlation':<12} {'Direction'}")
    print("-" * 70)
    
    for i, (feature, corr) in enumerate(correlations.head(20).items(), 1):
        # Get actual correlation (not absolute) to show direction
        actual_corr = X_clean[feature].corr(y)
        direction = "📈 Positive" if actual_corr > 0 else "📉 Negative"
        
        print(f"{i:<5} {feature:<40} {abs(actual_corr):<12.4f} {direction}")
    
    # Summary statistics
    print(f"\n📊 Correlation Analysis Summary:")
    print(f"  📈 Positive correlations: {(correlations > 0).sum()}")
    print(f"  📉 Negative correlations: {(correlations < 0).sum()}")
    print(f"  📊 Mean absolute correlation: {correlations.mean():.4f}")
    print(f"  📊 Max absolute correlation: {correlations.max():.4f}")
    print(f"  📊 Min absolute correlation: {correlations.min():.4f}")
    
    # Features with very low correlation
    low_corr_features = correlations[correlations < 0.01]
    if len(low_corr_features) > 0:
        print(f"\n⚠️  Features with very low correlation (< 0.01): {len(low_corr_features)}")
        print("   These features may not be useful for prediction.")
    
    return correlations


def main():
    """
    Main function to run the complete SEC data analysis and ML pipeline.
    
    This function orchestrates the entire workflow:
    1. Prepares and joins financial data with stock price trends
    2. Performs correlation analysis to identify predictive features
    3. Splits data by company (CIK) to prevent data leakage
    4. Tests multiple ML approaches with different missing value handling
    5. Compares performance and identifies the best model
    
    Returns:
        dict: Results dictionary containing performance metrics for all tested approaches
    """
    print("🚀 Starting SEC Data Analysis and ML Pipeline")
    print("=" * 60)
    
    # Prepare data
    X_train, X_val, y_train, y_val, feature_cols = prepare_data_for_model()
 
    # Build and compare models
    print(f"\n" + "="*60)
    print("Building and Comparing ML Models...")
    results = build_baseline_model(X_train, X_val, y_train, y_val, feature_cols)


if __name__ == "__main__":
    results = main()