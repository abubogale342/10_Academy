"""
Prediction module for credit risk assessment.
"""
import pandas as pd
import joblib
from typing import Dict, Any, Union

class CreditRiskPredictor:
    """
    A class to handle credit risk predictions using a trained model.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = joblib.load(model_path)
        self.feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None
    
    def preprocess_input(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Input data as dictionary or DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        # Ensure correct feature order if feature names are available
        if self.feature_names is not None:
            missing_cols = [col for col in self.feature_names if col not in df.columns]
            if missing_cols:
                for col in missing_cols:
                    df[col] = 0  # Fill missing columns with 0 or appropriate default
            df = df[self.feature_names]
            
        return df
    
    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data as dictionary or DataFrame
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        # Preprocess input
        df = self.preprocess_input(input_data)
        
        # Make predictions
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'prediction': int(pred),
                'probability_default': float(prob[1]),  # Probability of default (class 1)
                'probability_non_default': float(prob[0])  # Probability of non-default (class 0)
            })
            
        return {
            'predictions': results[0] if isinstance(input_data, dict) else results,
            'model_version': '1.0.0'  # You might want to load this from a config
        }

def load_predictor(model_path: str) -> CreditRiskPredictor:
    """
    Load a trained predictor.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        Initialized CreditRiskPredictor instance
    """
    return CreditRiskPredictor(model_path)

if __name__ == '__main__':
    # Example usage
    model_path = 'models/credit_risk_model.joblib'  # Update this path
    predictor = load_predictor(model_path)
    
    # Example prediction
    sample_input = {
        'income': 50000,
        'debt': 10000,
        'credit_score': 720,
        'employment_length': 5
    }
    
    result = predictor.predict(sample_input)
    print("Prediction result:", result)
