"""
Model training module for credit risk assessment.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

from .data_processing import prepare_data

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def save_model(model, model_path: str):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def train_pipeline(data_path: str, model_save_path: str = 'models/credit_risk_model.joblib') -> dict:
    """
    Complete training pipeline.
    
    Args:
        data_path: Path to training data
        model_save_path: Path to save the trained model
        
    Returns:
        Dictionary containing training results and metrics
    """
    # Prepare data
    X, y = prepare_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, model_save_path)
    
    return {
        'model': model,
        'metrics': metrics,
        'model_path': model_save_path
    }

if __name__ == '__main__':
    # Example usage
    data_path = '../../data/processed/train_data.csv'  # Update this path
    results = train_pipeline(data_path)
    print(f"Model trained and saved to {results['model_path']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
