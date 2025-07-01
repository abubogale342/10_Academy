"""
Pydantic models for the credit risk API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CreditApplication(BaseModel):
    """
    Schema for credit application input.
    """
    income: float = Field(..., gt=0, description="Annual income in USD")
    debt: float = Field(..., ge=0, description="Total debt in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_length: float = Field(..., ge=0, description="Years of employment")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    loan_term: int = Field(..., gt=0, description="Loan term in months")
    
    class Config:
        schema_extra = {
            "example": {
                "income": 75000.0,
                "debt": 15000.0,
                "credit_score": 720,
                "employment_length": 3.5,
                "loan_amount": 25000.0,
                "loan_term": 36
            }
        }

class PredictionResult(BaseModel):
    """
    Schema for prediction results.
    """
    prediction: int = Field(..., description="1 for high risk, 0 for low risk")
    probability_default: float = Field(..., ge=0, le=1, description="Probability of default (0-1)")
    probability_non_default: float = Field(..., ge=0, le=1, description="Probability of non-default (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "probability_default": 0.15,
                "probability_non_default": 0.85
            }
        }

class PredictionResponse(BaseModel):
    """
    Schema for the API response.
    """
    success: bool
    prediction: PredictionResult
    model_version: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "prediction": {
                    "prediction": 0,
                    "probability_default": 0.15,
                    "probability_non_default": 0.85
                },
                "model_version": "1.0.0"
            }
        }

class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    """
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid input data",
                "details": {"field": "income", "issue": "must be greater than 0"}
            }
        }
