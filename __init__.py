""""

This package provides a comprehensive solution for analyzing and predicting customer churn in the telecommunications industry.

Main Features:
- Data loading and preprocessing
- Exploratory data analysis
- Feature engineering
- Modeling, model evaluation & comparison
- Churn prediction
- Feature importance
- Hyperparameter tuning
- Data visualization

Example usage:

    from telco_churn import ChurnPredictionModel
    model = ChurnPredictionModel(data=df, target_column="Churn")
    model.prepare_data()
    model.split_data()
    model.scale_data()
    model.train_and_evaluate_models()
    model.evaluate_logistic_regression()
    model.feature_importance()
    model.random_forest_analysis()
    model.churn_risk_scoring()
"""

from .eda import DataAnalyzer
from models.churn_model import ChurnPredictionModel

__all__ = ['DataAnalyzer', 'ChurnPredictionModel']
