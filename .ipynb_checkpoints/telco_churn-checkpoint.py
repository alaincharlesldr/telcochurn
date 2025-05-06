### DATA WRANGLING & EDA ###

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load dataset 
from reusable_code.load_huggingface_dataset import load_huggingface_dataset
telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")

# Inspect data

telco.head()
telco.info()
telco.describe()

telco.isna().sum()
telco.nunique()


# Dealing with missing values
cols_missing_values = telco.columns[telco.isna().sum() > 0]
print(cols_missing_values)

# Checking missing values patterns across Churned/Non-churned users
non_churned = telco[telco['Churn'] == 0]
churned = telco[telco['Churn'] == 1]

churned.info()

# Imputing string to categorical data

telco['Internet Type'].fillna('No Internet Type', inplace=True)
telco['Offer'].fillna('Regular Plan', inplace=True)


# Checking again for missing value, should be non-existent
telco.isna().sum()

# data types

telco.dtypes

as_string = ['Customer ID', 'Zip Code']
telco[as_string] = telco[as_string].astype(str) 
as_category = ['Churn Category', 'Churn Reason','City', 'Country', 'Customer Status', 'Internet Type', 'Offer', 'Payment Method', 'Quarter', 'State']
telco[as_category] = telco[as_category].astype('category')
telco['Gender'] = telco['Gender'].map({'Male': 1, 'Female': 0})

telco.dtypes

# Exploratory data analysis

# Initial summary data

churned['Tenure in Months'].describe()
non_churned['Tenure in Months'].describe()

churned['Satisfaction Score'].describe()
non_churned['Satisfaction Score'].describe()

telco['Revenue / Tenure in Months'] = telco['Total Revenue'] / telco['Tenure in Months'] # creating new ratio feature 
churn_by_revenue_month = telco.groupby('Churn')['Revenue / Tenure in Months'].median()

telco['Recent Joiner'] = (telco['Tenure in Months'] < 7).astype('int') # creating new recent joiner feature
churn_by_recent_joiner = telco.groupby('Churn')['Recent Joiner'].value_counts()

age_labels = ['18-24','25-34', '35-44', '45-54', '+55'] # Creating age groups
age_ranges = [0, 25, 35, 45, 55, telco['Age'].max()]
telco['Age_levels']= pd.cut(telco['Age'] , bins=age_ranges , labels=age_labels)

telco['Churn_label'] = telco['Churn'].map({0: 'Active Users', 1: 'Churned Users'}) # adding clear labels
telco['Recent_Joiner_label'] = telco['Recent Joiner'].map({0: 'More than 6 months', 1: 'Less 6 months'})


# Creating variables subgroups

product_features = ['Avg Monthly GB Download', 'Contract', 'Device Protection Plan', 'Internet Type', 'Multiple Lines', 'Offer', 'Satisfaction Score', 'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Unlimited Data']

customer_features = ['Age', 'Age_levels', 'Customer Status', 'Gender', 'Number of Dependents', 'Number of Referrals', 'Payment Method', 'Premium Tech Support', 'Referred a Friend']

business_features = ['CLTV', 'Monthly Charge','Tenure in Months', 'Total Charges', 'Total Extra Data Charges', 'Total Revenue']


# Resuable functions

colors = ['#6a5acd', '#f67280']

# Defining a reusable plot_churn_stacked_bar function

def plot_churn_stacked_bar(crosstab_var, title, subtitle, xlabel):
    # Crosstab + normalization
    g = pd.crosstab(telco[crosstab_var], telco['Churn'], normalize='index') * 100
    g = g[[0, 1]]  # Ensure order: Active first, Churned second

    # Colors and plot
    colors = ['#6a5acd', '#f67280']
    ax = g.plot(kind='bar', stacked=True, color=colors, figsize=(8,6), edgecolor='black')

    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10, color='white', fontweight='bold')

    # Titles and labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.suptitle(subtitle, fontsize=10, color='gray')
    plt.ylabel("Percentage of Users", fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(['Active Users', 'Churned Users'], title='User Status')
    plt.tight_layout()
    plt.show()

# Defining a reusable plot_churn_box_chart function

def plot_churn_box_chart(ylabel, title, subtitle):
    
    # Colors and plot
    colors = ['#6a5acd', '#f67280']
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    g = sns.boxplot(data=telco, x='Churn_label', y=ylabel, palette=colors)

    # Titles and labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.suptitle(subtitle, fontsize=10, color='gray')
    plt.xlabel('', fontsize=12)
    plt.xticks(rotation=0)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()

# Defining a reusable plot_correlation_heatmap function

def plot_correlation_heatmap(data, title, subtitle, label_encode_cols=None, figsize=(10, 8)):
    """
    Plots a styled correlation heatmap with optional label encoding and upper triangle masking.

    Parameters:
    - data: DataFrame containing numeric and/or categorical columns
    - title: Main chart title (bold)
    - subtitle: Smaller, grey subheading for insight
    - label_encode_cols: list of categorical columns to encode (optional)
    - figsize: tuple, figure size
    - mask_upper: bool, mask upper triangle for symmetry (default: True)
    """
    
    df = data.copy()
    
    # Label encode categorical columns if specified
    if label_encode_cols:
        le = LabelEncoder()
        for col in label_encode_cols:
            df[col] = le.fit_transform(df[col])
    
    # Correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.set(style="white")
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        square=True
    )

    # Titles
    plt.title(title, fontsize=14, fontweight='bold')
    plt.suptitle(subtitle, fontsize=10, color='gray')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# Defining a reusable plot_kde_distribution function

def plot_kde_distribution(data, feature, title, subtitle, hue='Churn',
                          label_map={0: 'Active Users', 1: 'Churned Users'},
                          show_medians=True, xlim=None):
    """
    Plots a KDE distribution for a continuous variable by churn status (or other binary hue).

    Uses predefined color list 'colors' in the order [Active, Churned].

    Parameters:
    - data: DataFrame
    - feature: column to plot
    - title: main chart title (mandatory)
    - subtitle: smaller subheading (mandatory)
    - hue: binary column to separate groups
    - label_map: dict to rename hue labels
    - show_medians: whether to draw median lines
    - xlim: (min, max) x-axis range
    """

    df = data.copy()
    df['hue_label'] = df[hue].map(label_map)

    palette_dict = dict(zip(label_map.values(), colors))  # map your fixed colors list

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.kdeplot(
        data=df,
        x=feature,
        hue='hue_label',
        fill=True,
        common_norm=True,
        palette=palette_dict,
        alpha=0.4,
        linewidth=2,
        bw_adjust=1
    )

    if show_medians:
        for label, color in palette_dict.items():
            median_val = df.loc[df['hue_label'] == label, feature].median()
            plt.axvline(median_val, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            plt.text(median_val, plt.ylim()[1]*0.03, f'Median: {int(median_val)}',
                     color=color, ha='center', fontsize=9)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.suptitle(subtitle, fontsize=10, color='gray')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title='User Status', labels=['Churned Users', 'Active Users'])
    plt.tight_layout()
    plt.show()

# PRODUCT FEATURES

# By Contract Type
plot_churn_stacked_bar(
    crosstab_var='Contract',
    title="Churn Rate by Contract Type",
    subtitle="Users with long-term contracts churn significantly less",
    xlabel="Contract Type"
)

# By Offer
plot_churn_stacked_bar(
    crosstab_var='Offer',
    title="Churn Rate by Offer Type",
    subtitle="Offer A was great at retaining users, Offer E was terrible",
    xlabel="Offer Type"
)

# By Internet Type
plot_churn_stacked_bar(
    crosstab_var='Internet Type',
    title="Churn Rate by Internet Type",
    subtitle="Fiber Optic users churned the most — 'No/Missing Internet' users churn significantly less",
    xlabel="Internet Type"
)

# By Data Usage

plot_churn_box_chart(
    ylabel='Avg Monthly GB Download',
    title='Data Usage for Active vs. Churned Users',
    subtitle='The distribution of data usage of active and churned users overlaps largely'
)
    

# Churn & Satisfaction Score
plot_churn_box_chart(
    ylabel='Satisfaction Score',
    title='Churn vs Satisfaction Score',
    subtitle='Churn is happening at a Satisfaction Score below 3'
)

# Satisfaction Score across Offers



# Product features Heatmap

product_data = telco[['Avg Monthly GB Download', 'Contract', 'Device Protection Plan', 
                      'Internet Type', 'Multiple Lines', 'Offer', 
                      'Satisfaction Score', 'Streaming Movies', 
                      'Streaming Music', 'Streaming TV', 
                      'Unlimited Data', 'Churn']]

p_categorical_cols = ['Contract', 'Device Protection Plan', 'Internet Type', 'Multiple Lines', 'Offer', 
                    'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Unlimited Data']

plot_correlation_heatmap(
    data=product_data,
    title='Product Features Correlation with Churn',
    subtitle='Satisfaction Score is the strongest negative churn predictor — Contract type also stands out',
    label_encode_cols= p_categorical_cols
)
    

# CUSTOMER FEATURES

# By Gender
plot_churn_stacked_bar(
    crosstab_var='Gender',
    title="Churn Rate by Gender",
    subtitle="Churn is at similar levels across Male and Female usergroups",
    xlabel="Gender \n Female: O, Male: 1 "
)

# By Friend Referral
plot_churn_stacked_bar(
    crosstab_var='Referred a Friend',
    title="Churn Rate by Referral Activity",
    subtitle="Users that Referred a Friend are more sticky",
    xlabel="Referred a Friend? \n No: 0, Yes: 1"
)

# By Payment Method
plot_churn_stacked_bar(
    crosstab_var='Payment Method',
    title="Churn Rate by Payment Method",
    subtitle="Credit Card is associated with lower churn levels",
    xlabel="Payment Method"
)

# By Age
plot_churn_stacked_bar(
    crosstab_var='Age_levels',
    title="Churn Rate by Age",
    subtitle="Older age groups tend to have higher churn",
    xlabel="Age"
)


# Customer features heatmap
customer_data = telco[['Age', 'Age_levels', 'Gender', 'Number of Dependents', 'Number of Referrals', 'Payment Method', 'Premium Tech Support', 'Referred a Friend', 'Churn']].copy()

c_categorical_cols = ['Age_levels', 'Gender', 'Payment Method', 'Premium Tech Support', 
                    'Referred a Friend']

plot_correlation_heatmap(
    data=customer_data,
    title='Customer Features Correlation with Churn',
    subtitle='Referral Activity, Payment Method & Number of Dependents are all negatively correlated with Churn',
    label_encode_cols= c_categorical_cols
)


### BUSINESS FEATURES


# Tenure Distribution for Active & Churned groups
plot_kde_distribution(
    data=telco,
    feature='Tenure in Months',
    title="Tenure in Months Distribution by Churn",
    subtitle="Churn is happening in the first 6-12 months; Active users stay 4x longer",
    xlim=(0, None)
)


# Recent Joiners vs. Established users groups
# Create plot
plt.figure(figsize=(8, 6))
ax8 = sns.countplot(data=telco, x='Churn_label', hue='Recent_Joiner_label', palette=colors,edgecolor='black')

# Add labels on bars
for p in ax8.patches:
    height = p.get_height()
    ax8.annotate(f'{int(height)}', 
                (p.get_x() + p.get_width() / 2., height + 30), 
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='white')

# Title and subtitle
plt.title('Recent vs. Established Users by Churn Status', fontsize=14, fontweight='bold')
plt.suptitle("Nearly half of churned users had left within the first 6 months of tenure", fontsize=10, color='gray')

# Axis labels
plt.xlabel('User Churn Status', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Customize legend
plt.legend(title='User Tenure', title_fontsize=11, fontsize=10, loc='upper right', labels=['More than 6 months', 'Less than 6 months'])

# Final layout
sns.despine()
plt.tight_layout()
plt.show()

--


# Revenue & CLTV

# Revenue Distribution for Active & Churned groups
plot_kde_distribution(
    data=telco,
    feature='Total Revenue',
    title="Revenue Distribution across Active & Churned groups",
    subtitle="Every churned user represents revenue left on the table",
    xlim=(0, None)
)

# Customer Lifetime Value Distribution for Active & Churned groups
plot_kde_distribution(
    data=telco,
    feature='CLTV',
    title="Customer Lifetime Value Distribution across Active & Churned groups",
    subtitle="Churned users have significantly lower (c. 8-9% less) lifetime value than retained ones",
    xlim=(0, None)
)

# Revenue per Month for Active vs Churned Users
# Barplot on top for mean values
ax = sns.barplot(
    data=telco, 
    x='Churn_label', 
    y='Revenue / Tenure in Months', 
    palette=colors, 
    ci='sd', 
    edgecolor='black'
)

# Annotate exact values on bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}', 
                (p.get_x() + p.get_width() / 2., height + 1), 
                ha='center', fontsize=11, fontweight='bold', color='black')

# Titles and labels
plt.title('Revenue per Month: Active vs. Churned Users', fontsize=14, fontweight='bold')
plt.suptitle("Revenue per month is higher among churned users, hinting at aggressive short-term monetization", fontsize=10, color='gray')
plt.xlabel('')
plt.ylabel('Revenue / Month (€)', fontsize=12)
plt.tight_layout()
plt.show()


# Revenue & Extra Data Charges, across Active & Churned Users
sns.scatterplot(
    data=telco, 
    x='Total Revenue',
    y='Total Extra Data Charges', 
    hue='Churn_label', 
    palette=colors, 
    edgecolor='black')

# Titles and labels
plt.title('Revenue & Extra Data Charges: Active vs. Churned Users', fontsize=14, fontweight='bold')
plt.suptitle("A few users that churned were charged for extra data usage while not generating much revenue", fontsize=10, color='gray')
plt.xlabel('Revenue / Month (€)')
plt.ylabel('Total Extra Data Charges', fontsize=12)
plt.tight_layout()
plt.show()


# Business features heatmap

business_data = telco[['CLTV', 'Monthly Charge','Tenure in Months', 'Total Charges', 'Total Extra Data Charges', 'Total Revenue', 'Churn']].copy()

plot_correlation_heatmap(
    data=business_data,
    title='Business Features Correlation with Churn',
    subtitle='Churn is negatively correlated with Customer Lifetime Value and Revenues',
    label_encode_cols=None
)



---

# CHURN PREDICTION MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

# STEP 1: Define X1, X2, and y
product_features_w_score = ['Avg Monthly GB Download', 'Contract', 'Device Protection Plan', 'Internet Type',
                            'Multiple Lines', 'Offer', 'Satisfaction Score', 'Streaming Movies',
                            'Streaming Music', 'Streaming TV', 'Unlimited Data']

product_features_wo_score = [f for f in product_features_w_score if f != 'Satisfaction Score']

X1 = telco[product_features_w_score].copy()
X2 = telco[product_features_wo_score].copy()
y = telco['Churn']

# STEP 2: OneHot Encoding
from sklearn.preprocessing import OneHotEncoder

variables_to_encode = ['Contract', 'Internet Type', 'Offer']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_variables = pd.DataFrame(index=telco.index)

for var in variables_to_encode:
    transformed = ohe.fit_transform(telco[[var]])
    encoded_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out([var]), index=telco.index)
    encoded_variables = pd.concat([encoded_variables, encoded_df], axis=1)

# Append encoded features to X1 and X2
X1 = pd.concat([X1.drop(columns=variables_to_encode), encoded_variables], axis=1).astype('float64')
X2 = pd.concat([X2.drop(columns=variables_to_encode), encoded_variables], axis=1).astype('float64')

# Checking for missing values
X1.isna().sum()
X2.isna().sum()

# STEP 3: Split the data using StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X1, y):
    X1_train, X1_test = X1.iloc[train_idx], X1.iloc[test_idx]
    X2_train, X2_test = X2.iloc[train_idx], X2.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# STEP 4: Train the two models
model_with_score = LogisticRegression(max_iter=1000)
model_wo_score = LogisticRegression(max_iter=1000)

model_with_score.fit(X1_train, y_train)
model_wo_score.fit(X2_train, y_train)

# STEP 5: Predictions
y1_proba = model_with_score.predict_proba(X1_test)[:, 1]
y2_proba = model_wo_score.predict_proba(X2_test)[:, 1]

# STEP 6: ROC Curve
fpr1, tpr1, _ = roc_curve(y_test, y1_proba)
fpr2, tpr2, _ = roc_curve(y_test, y2_proba)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1, tpr1, label='With Satisfaction Score')
plt.plot(fpr2, tpr2, label='Without Satisfaction Score')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Churn Prediction')
plt.legend()
plt.tight_layout()
plt.show()

# STEP 7: Looking at model performance metrics (accuracy, sensitivity, precision)

# Predictions (class labels)
y1_pred = model_with_score.predict(X1_test)
y2_pred = model_wo_score.predict(X2_test)

    # CONFUSION MATRICES
print("📊 Confusion Matrix: With Satisfaction Score")
print(confusion_matrix(y_test, y1_pred))
print()

print("📊 Confusion Matrix: Without Satisfaction Score")
print(confusion_matrix(y_test, y2_pred))
print()

    # CLASSIFICATION REPORTS
print("🧾 Classification Report: With Satisfaction Score")
print(classification_report(y_test, y1_pred))
print()

print("🧾 Classification Report: Without Satisfaction Score")
print(classification_report(y_test, y2_pred))
print()

    # ROC-AUC Scores
auc_with = roc_auc_score(y_test, y1_proba)
auc_without = roc_auc_score(y_test, y2_proba)

print(f"ROC-AUC with Satisfaction Score: {auc_with:.4f}")
print(f"ROC-AUC without Satisfaction Score: {auc_without:.4f}")


# STEP 8: Feature importance 

importance1_df = pd.DataFrame({'Feature': X1_train.columns,'Coefficient': model_with_score.coef_[0]})
importance2_df = pd.DataFrame({'Feature': X2_train.columns,'Coefficient': w.coef_[0]})

importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='Coefficient', y='Feature', palette='coolwarm')
plt.title('Top 20 Feature Importances (Logistic Regression)', fontsize=14, fontweight='bold')
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logreg.coef_[0]  
})

importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='Coefficient', y='Feature', palette='coolwarm')
plt.title('Top 20 Feature Importances (Logistic Regression)', fontsize=14, fontweight='bold')
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()

