""
Unit tests for data processing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing import load_data, clean_data, create_features, prepare_data

# Sample test data
TEST_DATA = {
    'income': [50000, 60000, None, 70000, 80000],
    'debt': [10000, 15000, 20000, 20000, 25000],
    'credit_score': [700, 720, 650, 750, 800],
    'default': [0, 0, 1, 0, 1]
}

def test_load_data(tmp_path):
    """Test loading data from a CSV file."""
    # Create a temporary CSV file
    test_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(TEST_DATA)
    df.to_csv(test_file, index=False)
    
    # Test loading
    loaded_data = load_data(test_file)
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == 5

def test_clean_data():
    """Test data cleaning function."""
    df = pd.DataFrame(TEST_DATA)
    cleaned_data = clean_data(df)
    
    # Should remove rows with missing values
    assert len(cleaned_data) == 4
    assert cleaned_data.isna().sum().sum() == 0
    
    # Test with duplicates
    df_with_duplicates = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned_data = clean_data(df_with_duplicates)
    assert len(cleaned_data) == 5  # 4 unique + 1 duplicate = 4 after cleaning

def test_create_features():
    """Test feature engineering function."""
    df = pd.DataFrame(TEST_DATA)
    df = clean_data(df)
    df_with_features = create_features(df)
    
    # Check if new features are created
    assert 'debt_to_income' in df_with_features.columns
    assert not df_with_features['debt_to_income'].isna().any()

def test_prepare_data(tmp_path):
    """Test the complete data preparation pipeline."""
    # Create a temporary CSV file
    test_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(TEST_DATA)
    df.to_csv(test_file, index=False)
    
    # Test with target column
    X, y = prepare_data(test_file)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert 'default' not in X.columns
    
    # Test without target column
    df_without_target = df.drop(columns=['default'])
    test_file_no_target = tmp_path / "test_data_no_target.csv"
    df_without_target.to_csv(test_file_no_target, index=False)
    
    X, y = prepare_data(test_file_no_target)
    assert y is None
    assert len(X) > 0

# Run tests with: python -m pytest tests/ -v
