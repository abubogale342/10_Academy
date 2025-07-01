"""
Data processing module for credit risk model.
Handles data loading, cleaning, and feature engineering.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by handling missing values and outliers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Handle missing values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with new features
    """
    # Example feature engineering
    if 'income' in df.columns and 'debt' in df.columns:
        df['debt_to_income'] = df['debt'] / df['income']
    
    return df


def prepare_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Complete data preparation pipeline.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Tuple containing features and target (if target column exists)
    """
    df = load_data(file_path)
    df = clean_data(df)
    df = create_features(df)
    
    # If target column exists, separate it
    target_col = 'default'  # Adjust based on your data
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    
    return df, None
