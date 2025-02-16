import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, mean_squared_error, r2_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
import json
import optuna

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
    
    # Create binary labels based on rhythm
    # Now any rhythm other than 'Sinus Rhythm' is considered high risk
    rhythm_to_label = {
        'Sinus Rhythm': 0,  # Normal rhythm - low risk
        'Bradycardia': 0.8,  # High risk
        'Tachycardia': 0.85,  # High risk
        'Atrial Fibrillation': 0.9,  # Very high risk
        'Ventricular Tachycardia': 0.95,  # Very high risk
        'Ventricular Fibrillation': 1.0,  # Extreme risk
        'Asystole': 1.0,  # Extreme risk
        'Unknown': 0.7  # Default high risk for unknown rhythms
    }
    
    # Convert predicted_rhythm to risk scores
    df['label'] = df['predicted_rhythm'].map(rhythm_to_label)
    
    # Fill any NaN labels with high risk score (0.7)
    df['label'] = df['label'].fillna(0.7)
    
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
    print(df['label'].value_counts())
    print("\nRhythm distribution:")
    print(df['predicted_rhythm'].value_counts())
    
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
    labels = df['label']
    inputs = df.drop(columns=['label'])
    
    # Split the data without stratification
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.2, random_state=seed
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Define base parameters that won't be optimized
    base_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'feature_pre_filter': False
    }
    
    # Define the optimization objective
    def objective(trial):
        params = {
            **base_params,
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
        }
        
        # Train the model with current parameters
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
        )
        
        return model.best_score['valid_0']['l2']
    
    # Run the optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=opt_iters)
    
    # Train the final model with the best parameters
    best_params = {**base_params, **study.best_params}
    
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
    )
    
    # Calculate performance metrics
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'best_params': best_params
    }
    
    print("\nModel Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return final_model, metrics

def main():
    # Preprocess the data
    df = preprocess_data('vitals_100.csv')
    
    # Train the model
    model, metrics = train_model(df)
    
    # Print results
    print("\nModel Performance:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    
    # Save the model
    model.save_model('best_hrvarrest_model.txt')
    print("\nModel saved as 'best_hrvarrest_model.txt'")

if __name__ == "__main__":
    main()
