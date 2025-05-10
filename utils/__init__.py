"""Utility functions for Telco Customer Churn Analysis.

This package provides utility functions used throughout the Telco Customer Churn
Analysis project, including data loading and preprocessing utilities.

Main Features:
- Dataset loading from HuggingFace
- Data preprocessing utilities

Example usage:
    from utils.huggingface_loader import load_huggingface_dataset
    
    # Load dataset
    telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")
"""

from .huggingface_loader import load_huggingface_dataset

__all__ = ['load_huggingface_dataset']
