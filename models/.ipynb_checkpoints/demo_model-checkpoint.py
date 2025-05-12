"""Modeling demo for Telco Customer Churn Dataset.

This module demonstrates the use of the ChurnPredictionModel class for modeling customer churn using the Telco Customer Churn dataset.
"""

# Import ChurnPredictionModel
from models.churn_model import ChurnPredictionModel


def main():
    # Load dataset
    from utils.huggingface_loader import load_huggingface_dataset
    telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")
    
    # Initialize model
    model = ChurnPredictionModel(telco, 'Churn')
    
    # Prepare data
    model.prepare_data()
    
    # Split and scale data
    model.split_data()
    
    # Train and evaluate models
    model.train_and_evaluate_models()

    # Plot comparisons
    model._plot_comparisons()
    
    # Evaluate Logistic Regression
    model.evaluate_logistic_regression()
    
    # Analyze feature importance
    model.feature_importance()
    
    # Perform Random Forest analysis
    model.random_forest_analysis()
    
    # Perform hyperparameter tuning
    model.hyperparameter_tuning()
    
    # Calculate churn risk scores
    model.churn_risk_scoring()


if __name__ == "__main__":
    main()
