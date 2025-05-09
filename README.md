# Telco Customer Churn Analysis

A comprehensive analysis and prediction system for customer churn in the telecommunications industry.

## Overview

This project provides tools for analyzing customer churn patterns and predicting potential churners using machine learning. It includes:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training and Evaluation
- Churn Risk Scoring
- Visualization Tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alaincharlesldr/telcochurn.git
cd telcochurn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
telcochurn/
├── eda/
│   ├── data_analyzer.py    # Core EDA functionality
│   ├── demo_eda.py        # EDA demo script
│   └── __init__.py
├── models/
│   ├── churn_model.py     # Churn prediction model
│   ├── demo_model.py      # Modeling demo script
│   └── __init__.py
├── utils/
│   └── huggingface_loader.py  # Dataset loading utilities
├── requirements.txt
└── README.md
```

## Usage

### Exploratory Data Analysis

Run the EDA demo to analyze customer churn patterns across a range of customer, product & business variables:

```python
from eda import DataAnalyzer
from utils.huggingface_loader import load_huggingface_dataset

# Load dataset
telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")

# Initialize analyzer
analyzer = DataAnalyzer('aai510-group1/telco-customer-churn')

# Perform analysis
analyzer.exploratory_data_analysis()
analyzer.plot_churn_stacked_bar()
analyzer.plot_correlation_heatmap()
```

### Churn Prediction

Train and evaluate classification models to predict whether a user is likely to churn or not:

```python
from models.churn_model import ChurnPredictionModel

# Initialize model
model = ChurnPredictionModel(data=telco, target_column="Churn")

# Prepare and train
model.prepare_data()
model.split_data()
model.train_and_evaluate_models()

# Analyze results
model.feature_importance()
model.churn_risk_scoring()
```

## Features

- **Data Analysis**
  - Data cleaning
  - Exploratory data analysis
  - Correlation analysis
  - Data visualization

- **Modeling**
  - Data preprocessing
  - Classification models comparison
  - Feature importance analysis
  - Hyperparameter tuning
  - Churn risk scoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by HuggingFace
- Built with scikit-learn, pandas, seaborn and matplotlib
