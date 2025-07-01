"""
FastAPI application for credit risk prediction API.
"""
import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import joblib
import logging
from pathlib import Path

from .pydantic_models import (
    CreditApplication,
    PredictionResponse,
    ErrorResponse,
    PredictionResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk of loan applications",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "models/credit_risk_model.joblib")
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {str(e)}")
    model = None

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, responses={
    200: {"model": PredictionResponse},
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def predict(application: CreditApplication):
    """
    Make a prediction on a credit application.
    
    Args:
        application: Credit application data
        
    Returns:
        Prediction result with risk assessment
    """
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Model not loaded",
                details={"info": "The prediction model is not available"}
            ).dict()
        )
    
    try:
        # Prepare input data
        input_data = application.dict()
        
        # Make prediction (this is a simplified example)
        # In a real application, you would preprocess the input data
        # and use your actual model for prediction
        prediction = model.predict([list(input_data.values())])[0]
        probas = model.predict_proba([list(input_data.values())])[0]
        
        # Prepare response
        result = PredictionResult(
            prediction=int(prediction),
            probability_default=float(probas[1]),
            probability_non_default=float(probas[0])
        )
        
        return PredictionResponse(
            success=True,
            prediction=result,
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Prediction failed",
                details={"error": str(e)}
            ).dict()
        )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation error",
            details=exc.errors()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
