# Pytest tests for data_analyzer.py

import pytest
import pandas as pd
import numpy as np
from eda.data_analyzer import DataAnalyzer

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    data = {
        'Customer ID': ['1', '2', '3', '4', '5'],
        'Churn': [0, 1, 0, 1, 0],
        'Churn Category': ['No Churn', 'Competitor', 'No Churn', 'Price', 'No Churn'],
        'Churn Reason': ['None', 'Competitor made better offer', 'None', 'Price too high', 'None'],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'Country': ['USA', 'USA', 'USA', 'USA', 'USA'],
        'Customer Status': ['Active', 'Churned', 'Active', 'Churned', 'Active'],
        'Internet Type': ['Fiber Optic', 'DSL', 'Fiber Optic', 'DSL', 'Fiber Optic'],
        'Offer': ['Offer A', 'Offer B', 'Offer A', 'Offer B', 'Offer A'],
        'Payment Method': ['Credit Card', 'Bank Transfer', 'Credit Card', 'Bank Transfer', 'Credit Card'],
        'Quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q1'],
        'State': ['NY', 'CA', 'IL', 'TX', 'AZ'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Age': [25, 35, 45, 55, 65],
        'Tenure in Months': [12, 6, 24, 3, 36],
        'Satisfaction Score': [4, 2, 5, 1, 3],
        'Total Revenue': [1200, 600, 2400, 300, 3600],
        'CLTV': [1500, 750, 3000, 375, 4500],
        'Monthly Charge': [100, 100, 100, 100, 100],
        'Total Charges': [1200, 600, 2400, 300, 3600],
        'Total Extra Data Charges': [50, 25, 100, 12, 150],
        'Number of Dependents': [0, 1, 2, 0, 1],
        'Number of Referrals': [1, 0, 2, 0, 1],
        'Premium Tech Support': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Referred a Friend': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Zip Code': ['10001', '90001', '60601', '77001', '85001']
    }
    return pd.DataFrame(data)

@pytest.fixture
def analyzer(sample_data):
    """Create a DataAnalyzer instance with sample data."""
    analyzer = DataAnalyzer('test_dataset')
    analyzer.telco = sample_data
    return analyzer

def test_initialization():
    """Test DataAnalyzer initialization."""
    analyzer = DataAnalyzer('test_dataset')
    assert analyzer.telco is not None
    assert analyzer.colors == ['#6a5acd', '#f67280']
    assert analyzer.churned is None
    assert analyzer.non_churned is None
    assert analyzer.product_data is None
    assert analyzer.customer_data is None
    assert analyzer.business_data is None

def test_handle_missing_values(analyzer):
    """Test handling of missing values."""
    # Add some missing values
    analyzer.telco.loc[0, 'Internet Type'] = np.nan
    analyzer.telco.loc[1, 'Offer'] = np.nan
    
    analyzer.handle_missing_values()
    
    # Check if missing values are handled correctly
    assert analyzer.telco['Internet Type'].isna().sum() == 0
    assert analyzer.telco['Offer'].isna().sum() == 0
    assert 'No Internet Type' in analyzer.telco['Internet Type'].values
    assert 'Regular Plan' in analyzer.telco['Offer'].values

def test_set_data_types(analyzer):
    """Test data type conversion."""
    analyzer.set_data_types()
    
    # Check data types
    assert analyzer.telco['Customer ID'].dtype == 'object'
    assert analyzer.telco['Zip Code'].dtype == 'object'
    assert analyzer.telco['Churn Category'].dtype.name == 'category'
    assert analyzer.telco['Internet Type'].dtype.name == 'category'
    assert analyzer.telco['Gender'].dtype == 'int64'

def test_exploratory_data_analysis(analyzer):
    """Test exploratory data analysis."""
    analyzer.exploratory_data_analysis()
    
    # Check if subsets are created
    assert analyzer.churned is not None
    assert analyzer.non_churned is not None
    assert len(analyzer.churned) == 2  # 2 churned customers in sample data
    assert len(analyzer.non_churned) == 3  # 3 non-churned customers in sample data
    
    # Check if derived features are created
    assert 'Revenue / Tenure in Months' in analyzer.telco.columns
    assert 'Recent Joiner' in analyzer.telco.columns
    assert 'Age_levels' in analyzer.telco.columns
    assert 'Churn_label' in analyzer.telco.columns
    assert 'Recent_Joiner_label' in analyzer.telco.columns

def test_feature_groups(analyzer):
    """Test feature group definitions."""
    analyzer.exploratory_data_analysis()
    
    # Check product features
    assert analyzer.product_data is not None
    assert 'Avg Monthly GB Download' in analyzer.product_data.columns
    assert 'Contract' in analyzer.product_data.columns
    assert 'Churn' in analyzer.product_data.columns
    
    # Check customer features
    assert analyzer.customer_data is not None
    assert 'Age' in analyzer.customer_data.columns
    assert 'Gender' in analyzer.customer_data.columns
    assert 'Churn' in analyzer.customer_data.columns
    
    # Check business features
    assert analyzer.business_data is not None
    assert 'CLTV' in analyzer.business_data.columns
    assert 'Monthly Charge' in analyzer.business_data.columns
    assert 'Churn' in analyzer.business_data.columns

def test_categorical_columns(analyzer):
    """Test categorical column definitions."""
    analyzer.exploratory_data_analysis()
    
    # Check product categorical columns
    assert analyzer.p_categorical_cols is not None
    assert 'Contract' in analyzer.p_categorical_cols
    assert 'Internet Type' in analyzer.p_categorical_cols
    assert 'Offer' in analyzer.p_categorical_cols
    
    # Check customer categorical columns
    assert analyzer.c_categorical_cols is not None
    assert 'Age_levels' in analyzer.c_categorical_cols
    assert 'Gender' in analyzer.c_categorical_cols
    assert 'Payment Method' in analyzer.c_categorical_cols

def test_plot_churn_stacked_bar(analyzer):
    """Test churn stacked bar plot creation."""
    analyzer.exploratory_data_analysis()
    
    # Test plotting with different variables
    variables = ['Contract', 'Internet Type', 'Offer', 'Gender', 'Payment Method']
    for var in variables:
        try:
            analyzer.plot_churn_stacked_bar(
                crosstab_var=var,
                title=f"Churn Rate by {var}",
                subtitle=f"Test subtitle for {var}",
                xlabel=var
            )
        except Exception as e:
            pytest.fail(f"plot_churn_stacked_bar failed for {var}: {str(e)}")

def test_plot_churn_box_chart(analyzer):
    """Test churn box plot creation."""
    analyzer.exploratory_data_analysis()
    
    # Test plotting with different variables
    variables = ['Satisfaction Score', 'Tenure in Months', 'Monthly Charge']
    for var in variables:
        try:
            analyzer.plot_churn_box_chart(
                ylabel=var,
                title=f"Churn vs {var}",
                subtitle=f"Test subtitle for {var}"
            )
        except Exception as e:
            pytest.fail(f"plot_churn_box_chart failed for {var}: {str(e)}")

def test_plot_correlation_heatmap(analyzer):
    """Test correlation heatmap creation."""
    analyzer.exploratory_data_analysis()
    
    # Test plotting for different feature groups
    feature_groups = [
        (analyzer.product_data, analyzer.p_categorical_cols, "Product Features"),
        (analyzer.customer_data, analyzer.c_categorical_cols, "Customer Features"),
        (analyzer.business_data, None, "Business Features")
    ]
    
    for data, cat_cols, title in feature_groups:
        try:
            analyzer.plot_correlation_heatmap(
                data=data,
                title=f"{title} Correlation with Churn",
                subtitle=f"Test subtitle for {title}",
                label_encode_cols=cat_cols
            )
        except Exception as e:
            pytest.fail(f"plot_correlation_heatmap failed for {title}: {str(e)}")

def test_plot_kde_distribution(analyzer):
    """Test KDE distribution plot creation."""
    analyzer.exploratory_data_analysis()
    
    # Test plotting with different features
    features = ['Tenure in Months', 'Total Revenue', 'CLTV']
    for feature in features:
        try:
            analyzer.plot_kde_distribution(
                feature=feature,
                title=f"{feature} Distribution by Churn",
                subtitle=f"Test subtitle for {feature}",
                xlim=(0, None)
            )
        except Exception as e:
            pytest.fail(f"plot_kde_distribution failed for {feature}: {str(e)}")

def test_plot_revenue_per_month(analyzer):
    """Test revenue per month plot creation."""
    analyzer.exploratory_data_analysis()
    try:
        analyzer.plot_revenue_per_month()
    except Exception as e:
        pytest.fail(f"plot_revenue_per_month failed: {str(e)}")

def test_plot_revenue_extra_data_charges(analyzer):
    """Test revenue extra data charges plot creation."""
    analyzer.exploratory_data_analysis()
    try:
        analyzer.plot_revenue_extra_data_charges()
    except Exception as e:
        pytest.fail(f"plot_revenue_extra_data_charges failed: {str(e)}")

def test_plot_recent_vs_established_users(analyzer):
    """Test recent vs established users plot creation."""
    analyzer.exploratory_data_analysis()
    try:
        analyzer.plot_recent_vs_established_users()
    except Exception as e:
        pytest.fail(f"plot_recent_vs_established_users failed: {str(e)}")

def test_integration():
    """Test the complete pipeline with real data."""
    try:
        # Initialize analyzer with real dataset
        analyzer = DataAnalyzer('aai510-group1/telco-customer-churn')
        
        # Run through the complete analysis pipeline
        analyzer.inspect_data()
        analyzer.handle_missing_values()
        analyzer.set_data_types()
        analyzer.exploratory_data_analysis()
        
        # Test plotting functions
        analyzer.plot_churn_stacked_bar(
            crosstab_var='Contract',
            title="Churn Rate by Contract Type",
            subtitle="Test subtitle",
            xlabel="Contract Type"
        )
        
        analyzer.plot_churn_box_chart(
            ylabel='Satisfaction Score',
            title='Churn vs Satisfaction Score',
            subtitle='Test subtitle'
        )
        
        analyzer.plot_correlation_heatmap(
            data=analyzer.product_data,
            title='Product Features Correlation with Churn',
            subtitle='Test subtitle',
            label_encode_cols=analyzer.p_categorical_cols
        )
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {str(e)}")

