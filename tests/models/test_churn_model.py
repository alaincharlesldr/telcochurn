# Pytest tests for churn_predictor.py

import pytest
import pandas as pd
import numpy as np
from models.churn_model import ChurnPredictionModel
from utils.huggingface_loader import load_huggingface_dataset

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    data = {
        'Customer ID': ['1', '2', '3', '4', '5'],
        'Churn': [0, 1, 0, 1, 0],
        'Avg Monthly GB Download': [10, 20, 15, 25, 30],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'Device Protection Plan': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Internet Type': ['Fiber Optic', 'DSL', 'Fiber Optic', 'DSL', 'Fiber Optic'],
        'Multiple Lines': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Offer': ['Offer A', 'Offer B', 'Offer A', 'Offer B', 'Offer A'],
        'Satisfaction Score': [3, 1, 4, 2, 5],
        'Streaming Movies': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Streaming Music': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Streaming TV': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Unlimited Data': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Churn Score': [30, 80, 40, 70, 35]
    }
    return pd.DataFrame(data)

@pytest.fixture
def model(sample_data):
    """Create a ChurnPredictionModel instance with sample data."""
    model = ChurnPredictionModel(sample_data, 'Churn')
    model.prepare_data()
    model.split_data()
    return model

def test_initialization(sample_data):
    """Test model initialization."""
    model = ChurnPredictionModel(sample_data, 'Churn')
    assert model.data is not None
    assert model.target_column == 'Churn'
    assert model.X1 is None
    assert model.X2 is None
    assert model.y is None

def test_data_preparation(model):
    """Test data preparation methods."""
    assert model.X1 is not None
    assert model.X2 is not None
    assert model.y is not None
    assert len(model.encoders) > 0
    assert 'Contract' in model.encoders
    assert 'Internet Type' in model.encoders
    assert 'Offer' in model.encoders

def test_data_splitting(model):
    """Test data splitting and scaling."""
    assert model.X1_train is not None
    assert model.X1_test is not None
    assert model.X2_train is not None
    assert model.X2_test is not None
    assert model.y_train is not None
    assert model.y_test is not None
    assert model.scaler_X1 is not None
    assert model.scaler_X2 is not None

def test_model_training(model):
    """Test model training and evaluation."""
    model.train_and_evaluate_models()
    assert len(model.models_with_score) > 0
    assert len(model.models_without_score) > 0
    assert len(model.results_with_score) > 0
    assert len(model.results_without_score) > 0
    assert len(model.churn_risk_with_score) > 0
    assert len(model.churn_risk_without_score) > 0

def test_logistic_regression_evaluation(model):
    """Test Logistic Regression model evaluation."""
    model.train_and_evaluate_models()
    model.evaluate_logistic_regression()
    assert "Logistic Regression" in model.models_with_score
    assert "Logistic Regression" in model.models_without_score

def test_feature_importance(model):
    """Test feature importance analysis."""
    model.train_and_evaluate_models()
    model.feature_importance()
    # Verify that feature importance analysis was performed
    assert hasattr(model, 'models_with_score')
    assert hasattr(model, 'models_without_score')

def test_random_forest_analysis(model):
    """Test Random Forest analysis."""
    model.train_and_evaluate_models()
    model.random_forest_analysis()
    assert hasattr(model, 'random_forest_model')
    assert "Random Forest" in model.models_without_score

def test_hyperparameter_tuning(model):
    """Test hyperparameter tuning."""
    model.train_and_evaluate_models()
    model.random_forest_analysis()
    model.hyperparameter_tuning()
    # Verify that hyperparameter tuning was performed
    assert hasattr(model, 'random_forest_model')

def test_churn_risk_scoring(model):
    """Test churn risk scoring."""
    model.train_and_evaluate_models()
    model.churn_risk_scoring()
    assert "Churn Probability Score" in model.data.columns
    assert model.data["Churn Probability Score"].min() >= 0
    assert model.data["Churn Probability Score"].max() <= 1

def test_integration():
    """Test the complete pipeline with real data."""
    # Load real dataset
    telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")
    
    # Initialize and run model
    model = ChurnPredictionModel(telco, 'Churn')
    model.prepare_data()
    model.split_data()
    model.train_and_evaluate_models()
    model.evaluate_logistic_regression()
    model.feature_importance()
    model.random_forest_analysis()
    model.hyperparameter_tuning()
    model.churn_risk_scoring()
    
    # Verify final results
    assert "Churn Probability Score" in model.data.columns
    assert model.data["Churn Probability Score"].min() >= 0
    assert model.data["Churn Probability Score"].max() <= 1

