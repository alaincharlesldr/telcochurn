# Exploratory Data Analysis

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import DataAnalyzer
from eda.data_analyzer import DataAnalyzer

# Load & import dataset
from utils.huggingface_loader import load_huggingface_dataset
telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")


# Initialize DataAnalyzer
analyzer = DataAnalyzer('aai510-group1/telco-customer-churn')

# Data wrangling & EDA
analyzer.inspect_data()
analyzer.handle_missing_values()
analyzer.set_data_types()
analyzer.exploratory_data_analysis()

### PRODUCT FEATURES

# By Contract Type
analyzer.plot_churn_stacked_bar(
    crosstab_var='Contract',
    title="Churn Rate by Contract Type",
    subtitle="Users with long-term contracts churn significantly less",
    xlabel="Contract Type"
)

# By Offer
analyzer.plot_churn_stacked_bar(
    crosstab_var='Offer',
    title="Churn Rate by Offer Type",
    subtitle="Offer A was great at retaining users, Offer E was terrible",
    xlabel="Offer Type"
)

# By Internet Type
analyzer.plot_churn_stacked_bar(
    crosstab_var='Internet Type',
    title="Churn Rate by Internet Type",
    subtitle="Fiber Optic users churned the most — 'No/Missing Internet' users churn significantly less",
    xlabel="Internet Type"
)

# By Data Usage

analyzer.plot_churn_box_chart(
    ylabel='Avg Monthly GB Download',
    title='Data Usage for Active vs. Churned Users',
    subtitle='The distribution of data usage of active and churned users overlaps largely'
)
    

# Churn & Satisfaction Score
analyzer.plot_churn_box_chart(
    ylabel='Satisfaction Score',
    title='Churn vs Satisfaction Score',
    subtitle='Churn is happening at a Satisfaction Score below 3'
)

# Product Features Heatmap

analyzer.plot_correlation_heatmap(
    data=analyzer.product_data,
    title='Product Features Correlation with Churn',
    subtitle='Satisfaction Score is strongly negatively correlated with churn — Contract type also stands out',
    label_encode_cols=analyzer.p_categorical_cols
)

### CUSTOMER FEATURES

# By Gender
analyzer.plot_churn_stacked_bar(
    crosstab_var='Gender',
    title="Churn Rate by Gender",
    subtitle="Churn is at similar levels across Male and Female usergroups",
    xlabel="Gender \n Female: O, Male: 1 "
)

# By Friend Referral
analyzer.plot_churn_stacked_bar(
    crosstab_var='Referred a Friend',
    title="Churn Rate by Referral Activity",
    subtitle="Users that Referred a Friend are more sticky",
    xlabel="Referred a Friend? \n No: 0, Yes: 1"
)

# By Payment Method
analyzer.plot_churn_stacked_bar(
    crosstab_var='Payment Method',
    title="Churn Rate by Payment Method",
    subtitle="Credit Card is associated with lower churn levels",
    xlabel="Payment Method"
)

# By Age
analyzer.plot_churn_stacked_bar(
    crosstab_var='Age_levels',
    title="Churn Rate by Age",
    subtitle="Older age groups tend to have higher churn",
    xlabel="Age"
)

# Customer Features Heatmap
analyzer.plot_correlation_heatmap(
    data=analyzer.customer_data,
    title='Customer Features Correlation with Churn',
    subtitle='Satisfaction Score is the strongest negative churn predictor — Contract type also stands out',
    label_encode_cols=analyzer.c_categorical_cols
)

### BUSINESS FEATURES

# Tenure Distribution for Active & Churned groups
analyzer.plot_kde_distribution(
    feature='Tenure in Months',
    title="Tenure in Months Distribution by Churn",
    subtitle="Churn is happening in the first 6-12 months; Active users stay 4x longer",
    xlim=(0, None)
)


# Recent Joiners vs. Established users groups 
analyzer.plot_recent_vs_established_users()

# Revenue per Month
analyzer.plot_revenue_per_month()

# Revenue per Month (Extra Data Charges)
analyzer.plot_revenue_extra_data_charges()

# Revenue Distribution for Active & Churned groups
analyzer.plot_kde_distribution(
    feature='Total Revenue',
    title="Revenue Distribution across Active & Churned groups",
    subtitle="Every churned user represents revenue left on the table",
    xlim=(0, None)
)

# Customer Lifetime Value Distribution for Active & Churned groups
analyzer.plot_kde_distribution(
    feature='CLTV',
    title="Customer Lifetime Value Distribution across Active & Churned groups",
    subtitle="Churned users have significantly lower (c. 8-9% less) lifetime value than retained ones",
    xlim=(0, None)
)

# Business Features Heatmap

analyzer.plot_correlation_heatmap(
    data=analyzer.business_data,
    title='Business Features Correlation with Churn',
    subtitle='Churn is negatively correlated with Customer Lifetime Value and Revenues',
    label_encode_cols=None
)

