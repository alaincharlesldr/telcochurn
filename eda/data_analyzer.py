import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataAnalyzer:
    def __init__(self, dataset_path):
        self.telco = self.load_dataset(dataset_path)
        self.colors = ['#6a5acd', '#f67280']

    def load_dataset(self, dataset_path):
        from utils.huggingface_loader import load_huggingface_dataset
        return load_huggingface_dataset(dataset_path)

    def inspect_data(self):
        print(self.telco.head())
        self.telco.info()
        print(self.telco.describe())
        print(self.telco.isna().sum())
        print(self.telco.nunique())

    def handle_missing_values(self):
        cols_missing_values = self.telco.columns[self.telco.isna().sum() > 0]
        print(cols_missing_values)

        non_churned = self.telco[self.telco['Churn'] == 0]
        churned = self.telco[self.telco['Churn'] == 1]
        churned.info()

        self.telco['Internet Type'].fillna('No Internet Type', inplace=True)
        self.telco['Offer'].fillna('Regular Plan', inplace=True)
        print(self.telco.isna().sum())

    def set_data_types(self):
        self.telco[['Customer ID', 'Zip Code']] = self.telco[['Customer ID', 'Zip Code']].astype(str)
        as_category = ['Churn Category', 'Churn Reason', 'City', 'Country', 'Customer Status', 'Internet Type', 'Offer', 'Payment Method', 'Quarter', 'State']
        self.telco[as_category] = self.telco[as_category].astype('category')
        self.telco['Gender'] = self.telco['Gender'].map({'Male': 1, 'Female': 0})
        print(self.telco.dtypes)

    def exploratory_data_analysis(self):
        self.churned = self.telco[self.telco['Churn'] == 1]
        self.non_churned = self.telco[self.telco['Churn'] == 0]

        print(self.churned['Tenure in Months'].describe())
        print(self.non_churned['Tenure in Months'].describe())
        print(self.churned['Satisfaction Score'].describe())
        print(self.non_churned['Satisfaction Score'].describe())

        self.telco['Revenue / Tenure in Months'] = self.telco['Total Revenue'] / self.telco['Tenure in Months']
        churn_by_revenue_month = self.telco.groupby('Churn')['Revenue / Tenure in Months'].median()

        self.telco['Recent Joiner'] = (self.telco['Tenure in Months'] < 7).astype('int')
        churn_by_recent_joiner = self.telco.groupby('Churn')['Recent Joiner'].value_counts()

        age_labels = ['18-24', '25-34', '35-44', '45-54', '+55']
        age_ranges = [0, 25, 35, 45, 55, self.telco['Age'].max()]

        self.telco['Age_levels'] = pd.cut(self.telco['Age'], bins=age_ranges, labels=age_labels)
        self.age_levels = self.telco['Age_levels']  

        self.telco['Churn_label'] = self.telco['Churn'].map({0: 'Active Users', 1: 'Churned Users'})
        self.telco['Recent_Joiner_label'] = self.telco['Recent Joiner'].map({0: 'More than 6 months', 1: 'Less 6 months'})

        # Define product, customer & business data
        self.product_data = self.telco[['Avg Monthly GB Download', 'Contract', 'Device Protection Plan',
                      'Internet Type', 'Multiple Lines', 'Offer',
                      'Satisfaction Score', 'Streaming Movies',
                      'Streaming Music', 'Streaming TV',
                      'Unlimited Data', 'Churn']].copy()

        self.p_categorical_cols = ['Contract', 'Device Protection Plan', 'Internet Type', 'Multiple Lines', 'Offer',
                    'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Unlimited Data']

        
        self.customer_data = self.telco[['Age', 'Age_levels', 'Gender', 'Number of Dependents', 'Number of Referrals', 'Payment Method', 'Premium Tech Support', 'Referred a Friend', 'Churn']].copy()

        self.c_categorical_cols = ['Age_levels', 'Gender', 'Payment Method', 'Premium Tech Support', 
                    'Referred a Friend']

        self.business_data = self.telco[['CLTV', 'Monthly Charge','Tenure in Months', 'Total Charges', 'Total Extra Data Charges', 'Total Revenue', 'Churn']].copy()


    def plot_churn_stacked_bar(self, crosstab_var, title, subtitle, xlabel):
        g = pd.crosstab(self.telco[crosstab_var], self.telco['Churn'], normalize='index') * 100
        g = g[[0, 1]]

        ax = g.plot(kind='bar', stacked=True, color=self.colors, figsize=(8, 6), edgecolor='black')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10, color='white', fontweight='bold')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.ylabel("Percentage of Users", fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(['Active Users', 'Churned Users'], title='User Status')
        plt.tight_layout()
        plt.show()

    def plot_churn_box_chart(self, ylabel, title, subtitle):
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")

        g = sns.boxplot(data=self.telco, x='Churn_label', y=ylabel, palette=self.colors)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.xlabel('', fontsize=12)
        plt.xticks(rotation=0)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, data, title, subtitle, label_encode_cols=None, figsize=(10, 8)):
        df = data.copy()

        if label_encode_cols:
            le = LabelEncoder()
            for col in label_encode_cols:
                df[col] = le.fit_transform(df[col])

        corr_matrix = df.corr()

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

        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_kde_distribution(self, feature, title, subtitle, hue='Churn',
                              label_map={0: 'Active Users', 1: 'Churned Users'},
                              show_medians=True, xlim=None):
        df = self.telco.copy()
        df['hue_label'] = df[hue].map(label_map)

        palette_dict = dict(zip(label_map.values(), self.colors))

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
                plt.text(median_val, plt.ylim()[1] * 0.03, f'Median: {int(median_val)}',
                         color=color, ha='center', fontsize=9)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title='User Status', labels=['Churned Users', 'Active Users'])
        plt.tight_layout()
        plt.show()

    def plot_recent_vs_established_users(self):
        plt.figure(figsize=(8, 6))
        ax8 = sns.countplot(data=self.telco, x='Churn_label', hue='Recent_Joiner_label', palette=self.colors, edgecolor='black')

        for p in ax8.patches:
            height = p.get_height()
            ax8.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height + 30),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='white')

        plt.title('Recent vs. Established Users by Churn Status', fontsize=14, fontweight='bold')
        plt.suptitle("Nearly half of churned users had left within the first 6 months of tenure", fontsize=10, color='gray')
        plt.xlabel('User Churn Status', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(title='User Tenure', title_fontsize=11, fontsize=10, loc='upper right', labels=['More than 6 months', 'Less than 6 months'])
        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_revenue_per_month(self):
        ax = sns.barplot(
            data=self.telco,
            x='Churn_label',
            y='Revenue / Tenure in Months',
            palette=self.colors,
            ci='sd',
            edgecolor='black'
        )

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}',
                        (p.get_x() + p.get_width() / 2., height + 1),
                        ha='center', fontsize=11, fontweight='bold', color='black')

        plt.title('Revenue per Month: Active vs. Churned Users', fontsize=14, fontweight='bold')
        plt.suptitle("Revenue per month is higher among churned users, hinting at aggressive short-term monetization", fontsize=10, color='gray')
        plt.xlabel('')
        plt.ylabel('Revenue / Month (€)', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_revenue_extra_data_charges(self):
        sns.scatterplot(
            data=self.telco,
            x='Total Revenue',
            y='Total Extra Data Charges',
            hue='Churn_label',
            palette=self.colors,
            edgecolor='black')

        plt.title('Revenue & Extra Data Charges: Active vs. Churned Users', fontsize=14, fontweight='bold')
        plt.suptitle("A few users that churned were charged for extra data usage while not generating much revenue", fontsize=10, color='gray')
        plt.xlabel('Revenue / Month (€)')
        plt.ylabel('Total Extra Data Charges', fontsize=12)
        plt.tight_layout()
        plt.show()
