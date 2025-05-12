# ChurnModel - Streamlit Application

An interactive web application for analyzing and predicting customer churn in the telecommunications industry.

## Overview

The application provides a comprehensive interface for exploring customer churn data, analyzing patterns, and making predictions. It's built using Streamlit and combines data analysis, visualization, and machine learning capabilities.

## Features

### 🔸 Overview
- Key metrics dashboard:
  - Total Customers (4,225)
  - Ratio Churned (26.5%)
  - Average Satisfaction (3.2/5)
  - Average Revenue per user ($3,065.81)
  - Average Tenure (Active: 39 months, Churned: 10 months)
- Interactive visualizations:
  - Customer Churn Distribution
  - Satisfaction Score Distribution
  - Tenure Distribution
  - Revenue Distribution

### 📊 Data Analysis
Organized into three main sections:

#### Customer Features
- Demographics analysis (Gender, Age)
- Payment methods
- Referral activity
- Customer satisfaction patterns
- Customer features correlation heatmap

#### Product Features
- Contract types and durations
- Internet service types
- Additional services
- Data usage patterns
- Product features correlation heatmap

#### Business Features
- Revenue analysis
- Customer Lifetime Value (CLTV)
- Tenure patterns
- Revenue per month
- Extra data charges
- Business features correlation heatmap

### 🤖 Modeling
- Model Performance Comparison:
  - ROC-AUC scores
  - Cross-validation results
  - Churn probability distributions
- Feature Importance Analysis:
  - Logistic Regression coefficients
  - Random Forest feature importance
  - Permutation importance
- Model Comparison:
  - Logistic Regression
  - KNN
  - Decision Tree
  - Random Forest
- Hyperparameter Tuning Results

### 🎯 Simulator
- Single Customer Prediction:
  - Interactive form for customer features
  - Real-time churn probability
  - Top influencing factors
- Batch Prediction:
  - CSV file upload
  - Bulk predictions
  - Results download

## Technology Stack
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Frontend**: Streamlit
- **Version Control**: Git

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/alaincharlesldr/telcochurn.git
cd telcochurn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app/app.py
```

## Data Source
The dataset is sourced from Hugging Face and contains customer data from a telecommunications company.

## Version
This is version 1 of this project. Next versions will include other classification models (such as XGBoost) and dimensionality reduction.

## Author
Alain-Charles Lauriano do Rego
- [GitHub](https://github.com/alaincharlesldr/telcochurn)
- [Medium](https://medium.com/@alaincharlesldr/a-60-uplift-in-churn-recall-the-case-for-prioritizing-customer-surveys-9cb473e9685e)
- [LinkedIn](https://www.linkedin.com/in/alain-charles-lauriano-do-rego-7001b089/)
- Email: alaincharlesldr@gmail.com 