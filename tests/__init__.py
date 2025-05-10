"""Test suite for Telco Customer Churn Analysis.

This package contains test modules for the Telco Customer Churn Analysis project.
The tests are organized into subdirectories corresponding to the main package structure:

- eda/: Tests for exploratory data analysis functionality
- models/: Tests for churn prediction models
- utils/: Tests for utility functions

Example usage:
    pytest tests/  # Run all tests
    pytest tests/eda/  # Run only EDA tests
    pytest tests/models/  # Run only model tests
"""

from . import eda
from . import models
from . import utils

__all__ = ['eda', 'models', 'utils']
