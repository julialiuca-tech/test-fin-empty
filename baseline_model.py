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

COMPLETENESS_THRESHOLD = 0.2

featurized_data_file = 'processed_data/featurized_simplified.csv'

trend_horizon_in_months = 1
stock_trend_data_file = f'stock_data/price_trends_{trend_horizon_in_months}month.csv'
Y_LABEL = 'trend_5per_up' # can be 'trend_up_or_down' or 'trend_5per_up'
SPLIT_STRATEGY = 'random' # can be 'cik', 'date', or 'random'


def split_data(df, split_strategy=SPLIT_STRATEGY):
    """
    Split data into training and validation sets using different strategies.
    
    Args:
        df (pd.DataFrame): Full dataset with metadata columns
        split_strategy (str): Splitting strategy ('cik', 'date', or 'random')
        
    Returns:
        tuple: (train_mask, val_mask) - Boolean masks for training and validation sets
    """
    if split_strategy == 'cik':
        # Split by CIK (companies) - prevents data leakage
        print(f"\n🔄 Splitting data by CIK...")
        unique_ciks = df['cik'].unique()
        print(f"📊 Total unique companies (CIKs): {len(unique_ciks)}")
        
        # Split CIKs into train/validation sets
        train_ciks, val_ciks = train_test_split(unique_ciks, test_size=0.3, random_state=42)
        
        # Create train/validation masks based on CIK
        train_mask = df['cik'].isin(train_ciks)
        val_mask = df['cik'].isin(val_ciks)
        
        print(f"📊 Training set: {train_mask.sum()} samples from {len(train_ciks)} companies")
        print(f"📊 Validation set: {val_mask.sum()} samples from {len(val_ciks)} companies")
        
    elif split_strategy == 'date':
        # Split by date - use 70th percentile of data points for training
        print(f"\n🔄 Splitting data by date (70th percentile of data points for training)...")
        
        # Get unique year_month periods and sort them
        unique_periods = sorted(df['year_month'].unique())
        if len(unique_periods) < 2:
            print("❌ Need at least 2 periods for date-based splitting. Falling back to CIK splitting.")
            return split_data(df, 'cik')
        
        # Count data points per period and sort by count
        period_counts = df['year_month'].value_counts().sort_index()
        print(f"📊 Data points per period: {dict(period_counts)}")
        
        # Calculate cumulative data points and find 70th percentile cutoff
        cumulative_counts = period_counts.cumsum()
        total_samples = len(df)
        cutoff_point = int(total_samples * 0.8)
        
        # Find periods that fall within the 70th percentile
        train_periods = cumulative_counts[cumulative_counts <= cutoff_point].index.tolist()
        val_periods = cumulative_counts[cumulative_counts > cutoff_point].index.tolist()
        
        # Create train/validation masks based on year_month
        train_mask = df['year_month'].isin(train_periods)
        val_mask = df['year_month'].isin(val_periods)
        
        print(f"📊 Training: {train_mask.sum()} samples from {len(train_periods)} periods {[str(p) for p in train_periods]}")
        print(f"📊 Validation: {val_mask.sum()} samples from {len(val_periods)} periods {[str(p) for p in val_periods]}")
        
    elif split_strategy == 'random':
        # Split randomly - 70% for training, 30% for validation
        print(f"\n🔄 Splitting data randomly (70% training, 30% validation)...")
        
        # Get total number of samples
        total_samples = len(df)
        train_size = int(total_samples * 0.7)
        
        # Create random indices for training
        np.random.seed(42)  # For reproducibility
        train_indices = np.random.choice(total_samples, size=train_size, replace=False)
        
        # Create boolean masks
        train_mask = np.zeros(total_samples, dtype=bool)
        train_mask[train_indices] = True
        val_mask = ~train_mask
        
        print(f"📊 Training: {train_mask.sum()} samples ({train_mask.sum()/total_samples*100:.1f}%)")
        print(f"📊 Validation: {val_mask.sum()} samples ({val_mask.sum()/total_samples*100:.1f}%)")
        
    else:
        raise ValueError(f"Invalid split_strategy: {split_strategy}. Use 'cik', 'date', or 'random'.")
    
    return train_mask, val_mask


def prepare_data_for_model(split_strategy=SPLIT_STRATEGY):
    """
    Prepare and join featurized financial data with stock price trend data.
    
    This function loads the featurized financial data and stock price trend data,
    performs an inner join on CIK and year_month, and returns the prepared
    training and validation datasets.
    
    Args:
        split_strategy (str): Data splitting strategy ('cik', 'date', or 'random')
            - 'cik': Split by company (CIK) to prevent data leakage
            - 'date': Split by time (70th percentile of data points for training)
            - 'random': Random split (70% training, 30% validation)
    
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
    y = df[Y_LABEL].copy()
    
    # 2. Split samples into training and validation datasets
    train_mask, val_mask = split_data(df, split_strategy=split_strategy)
    
    # Apply masks to get training and validation sets
    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    
    return X_train, X_val, y_train, y_val, feature_cols


def correlation_analysis(df):
    """
    Perform correlation analysis between features and target variable.
    
    Args:
        df (pd.DataFrame): Dataset with features and target labels
        
    Returns:
        pd.DataFrame: Correlation results sorted by absolute correlation
    """
    # Identify feature columns
    feature_cols = [col for col in df.columns if '_current' in col or '_change' in col]
    
    if len(feature_cols) == 0:
        print("❌ No feature columns found for correlation analysis.")
        return pd.DataFrame()
    
    print(f"📊 Analyzing correlations for {len(feature_cols)} features...")
    
    # Fill missing values with median for correlation calculation
    df_filled = df[feature_cols + [Y_LABEL]].fillna(df[feature_cols].median())
    
    # Calculate correlations with target
    correlations = df_filled[feature_cols].corrwith(df_filled[Y_LABEL]).abs().sort_values(ascending=False)
    
    # Display top 20 features
    print(f"\n🔍 Top 20 Features by Absolute Correlation with {Y_LABEL}:")
    print("=" * 80)
    for i, (feature, corr) in enumerate(correlations.head(20).items(), 1):
        direction = "📈" if df_filled[feature].corr(df_filled[Y_LABEL]) > 0 else "📉"
        print(f"{i:2d}. {feature:<40} {direction} {corr:.4f}")
    
    # Summary statistics
    print(f"\n📊 Correlation Summary:")
    print(f"  Mean absolute correlation: {correlations.mean():.4f}")
    print(f"  Max absolute correlation: {correlations.max():.4f}")
    print(f"  Features with |corr| > 0.1: {(correlations > 0.1).sum()}")
    print(f"  Features with |corr| > 0.05: {(correlations > 0.05).sum()}")
    
    return correlations


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


def apply_imputation(X_train, X_val, imputation_strategy='none'): 
    if imputation_strategy == 'median':
        # Simple median imputation (baseline approach)
        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())  # Use training median for validation
        return X_train_imputed, X_val_imputed

    else: 
        return X_train.copy(), X_val.copy()


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


def build_baseline_model(X_train, X_val, y_train, y_val, feature_cols):
    """
    Build and evaluate baseline models for stock trend prediction with different missing value handling approaches.

    This function tests multiple approaches for handling missing values and model types:
    1. Random Forest with native missing value support
    2. XGBoost with native missing value support  
    3. Median imputation + Random Forest
    4. Current features only experiments

    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (pd.Series): Training target labels
        y_val (pd.Series): Validation target labels
        feature_cols (list): List of feature column names

    Returns:
        dict: Results dictionary containing performance metrics for all tested approaches
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
                     result = test_model(X_train_imputed, X_val_imputed, y_train, y_val, model_name)
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


def calculate_threshold_rates(y_true, y_pred_proba, threshold):
    """
    Calculate FPR and FNR at a specific threshold
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        dict: FPR, FNR, and other metrics
    """
    from sklearn.metrics import confusion_matrix
    
    # Create binary predictions
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    TN, FP, FN, TP = cm.ravel()
    
    # Calculate rates
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Sensitivity)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate (Specificity)
    
    # Additional metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return {
        'threshold': threshold,
        'FPR': FPR,
        'FNR': FNR,
        'TPR': TPR,
        'TNR': TNR,
        'precision': precision,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }


def analyze_thresholds(y_true, y_pred_proba, thresholds=[0.25, 0.5, 0.75, 0.9]):
    """Analyze performance at different thresholds"""
    
    results = []
    for threshold in thresholds:
        result = calculate_threshold_rates(y_true, y_pred_proba, threshold)
        results.append(result)
        
        # Calculate number of predicted positives
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        num_predicted_positives = y_pred_binary.sum()
        total_samples = len(y_pred_binary)
        
        print(f"\nThreshold: {threshold}")
        print(f"  Predicted Positives: {num_predicted_positives}/{total_samples} ({num_predicted_positives/total_samples*100:.1f}%)")
        print(f"  FPR: {result['FPR']:.4f} ({result['FPR']*100:.2f}%)")
        print(f"  FNR: {result['FNR']:.4f} ({result['FNR']*100:.2f}%)")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
    
    return results


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
    
    # Prepare data (you can change split_strategy to 'date' for time-based splitting)
    X_train, X_val, y_train, y_val, feature_cols = prepare_data_for_model(split_strategy=SPLIT_STRATEGY)
 
    # Build and compare models
    print(f"\n" + "="*60)
    print("Building and Comparing ML Models...")
    results = build_baseline_model(X_train, X_val, y_train, y_val, feature_cols)
    
    return results


if __name__ == "__main__":
    results = main()
