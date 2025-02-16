import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
import json

def preprocess_data(file_path='vitals_100.csv'):
    """
    Preprocess vital signs data for heart rhythm classification.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Select only numeric columns for features
    feature_columns = [
        'temperature', 'heartrate', 'resprate', 'o2sat', 
        'sbp', 'dbp', 'pain', 'hour', 'day_of_week',
        'hr_change', 'rr_change', 'o2_change'
    ]
    
    # Ensure all feature columns are numeric
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN values with 0
    df[feature_columns] = df[feature_columns].fillna(0)
    
    # Create binary labels from predicted_rhythm
    rhythm_to_label = {
        'Bradycardia': 0, 
        'Sinus Rhythm': 0,
        'Tachycardia': 1, 
        'Ventricular Tachycardia': 1,
        'Unknown': 0  # Adding Unknown mapping
    }
    
    # Convert predicted_rhythm to binary labels
    df['label'] = df['predicted_rhythm'].map(rhythm_to_label)
    
    # Save the processed data with the new label column
    output_path = file_path
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    # Keep only the features we want and the label for training
    df_train = df[feature_columns + ['label']]
    
    # Display the processed dataset info
    print("\nFirst few rows of the processed dataset:")
    print(df_train.head())
    print("\nDataset shape:", df_train.shape)
    print("\nLabel distribution:")
    print(df_train['label'].value_counts())
    print("\nFeature columns:", feature_columns)
    
    return df_train

def train_model(df, seed=42, opt_inits=3, opt_iters=50):
    """
    Train a LightGBM model on the preprocessed vital signs data.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe from preprocess_data()
        seed (int): Random seed for reproducibility
        opt_inits (int): Number of initial points for Bayesian optimization
        opt_iters (int): Number of optimization iterations
        
    Returns:
        lgb.Booster: Trained model
        dict: Performance metrics
    """
    # Prepare the data
    labels = df['label']  # Labels are already binary from preprocess_data
    
    # Remove label column
    inputs = df.drop(columns=['label'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    
    # Create LightGBM datasets
    dtrain = lgb.Dataset(X_train, y_train)
    dtest = lgb.Dataset(X_test, y_test)
    
    # Define parameter bounds for optimization
    param_bounds = {
        'num_leaves': (16, 32),
        'lambda_l1': (0.7, 0.9),
        'lambda_l2': (0.9, 1),
        'feature_fraction': (0.6, 0.7),
        'bagging_fraction': (0.6, 0.9),
        'min_child_samples': (6, 10),
        'min_child_weight': (10, 40)
    }
    
    # Fixed parameters
    fixed_params = {
        'objective': 'binary',
        'learning_rate': 0.005,
        'bagging_freq': 1,
        'force_row_wise': True,
        'max_depth': 5,
        'verbose': -1,
        'random_state': seed,
        'n_jobs': -1,
    }
    
    def auprc(preds, dtrain):
        labels = dtrain.get_label()
        return 'auprc', average_precision_score(labels, preds), True
    
    def eval_function(num_leaves, lambda_l1, lambda_l2, feature_fraction,
                     bagging_fraction, min_child_samples, min_child_weight):
        params = {
            'num_leaves': int(round(num_leaves)),
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'min_child_samples': int(round(min_child_samples)),
            'min_child_weight': min_child_weight,
            'feature_pre_filter': False,
        }
        params.update(fixed_params)
        
        # Create callbacks list for early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=300)]
        
        # Train model with current parameters
        model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=2000,
            valid_sets=[dtest],
            callbacks=callbacks,  # Use callbacks instead of early_stopping_rounds
            feval=auprc,
        )
        
        # Get predictions and score
        preds = model.predict(X_test)
        score = average_precision_score(y_test, preds)
        return score
    
    # Perform Bayesian optimization
    optimizer = BayesianOptimization(
        f=eval_function,
        pbounds=param_bounds,
        random_state=seed
    )
    
    optimizer.maximize(init_points=opt_inits, n_iter=opt_iters)
    
    # Get best parameters and train final model
    best_params = optimizer.max['params']
    best_params['num_leaves'] = int(round(best_params['num_leaves']))
    best_params['min_child_samples'] = int(round(best_params['min_child_samples']))
    best_params.update(fixed_params)
    
    # Create callbacks list for early stopping
    callbacks = [lgb.early_stopping(stopping_rounds=300)]
    
    final_model = lgb.train(
        params=best_params,
        train_set=dtrain,
        num_boost_round=2000,
        valid_sets=[dtest],
        callbacks=callbacks,  # Use callbacks instead of early_stopping_rounds
        feval=auprc
    )
    
    # Generate predictions and metrics
    y_prob = final_model.predict(X_test)
    metrics = {
        'auroc': roc_auc_score(y_test, y_prob),
        'auprc': average_precision_score(y_test, y_prob)
    }
    
    return final_model, metrics

def main():
    # Preprocess the data
    df = preprocess_data('vitals_100.csv')
    
    # Train the model
    model, metrics = train_model(df)
    
    # Print results
    print("\nModel Performance:")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    
    # Save the model
    model.save_model('best_hrvarrest_model.txt')
    print("\nModel saved as 'best_hrvarrest_model.txt'")

if __name__ == "__main__":
    main()
