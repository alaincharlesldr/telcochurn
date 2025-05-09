"""Data Analysis Module for Telco Customer Churn Dataset.

This module provides a comprehensive set of tools for analyzing and visualizing
customer churn data in the telecommunications industry.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Union, Any


class DataAnalyzer:
    """A class for analyzing and visualizing customer churn data.
    
    This class provides methods for data loading, preprocessing, and various
    types of visualizations to analyze customer churn patterns.
    
    Attributes:
        telco (pd.DataFrame): The main dataset containing customer information.
        colors (List[str]): Color palette for visualizations.
        churned (pd.DataFrame): Subset of data for churned customers.
        non_churned (pd.DataFrame): Subset of data for active customers.
        product_data (pd.DataFrame): Product-related features.
        customer_data (pd.DataFrame): Customer demographic features.
        business_data (pd.DataFrame): Business metrics features.
    """

    def __init__(self, dataset_path: str) -> None:
        """Initialize the DataAnalyzer.
        
        Args:
            dataset_path (str): Path to the dataset on HuggingFace.
        """
        self.telco = self.load_dataset(dataset_path)
        self.colors = ['#6a5acd', '#f67280']
        self.churned = None
        self.non_churned = None
        self.product_data = None
        self.customer_data = None
        self.business_data = None
        self.p_categorical_cols = None
        self.c_categorical_cols = None

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load the dataset from HuggingFace.
        
        Args:
            dataset_path (str): Path to the dataset on HuggingFace.
            
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        from utils.huggingface_loader import load_huggingface_dataset
        return load_huggingface_dataset(dataset_path)

    def inspect_data(self) -> None:
        """Display basic information about the dataset.
        
        Shows the first few rows, data types, summary statistics,
        missing values, and unique value counts.
        """
        print(self.telco.head())
        self.telco.info()
        print(self.telco.describe())
        print(self.telco.isna().sum())
        print(self.telco.nunique())

    def handle_missing_values(self) -> None:
        """Handle missing values in the dataset.
        
        Identifies columns with missing values and fills them with
        appropriate default values.
        """
        cols_missing_values = self.telco.columns[self.telco.isna().sum() > 0]
        print(cols_missing_values)

        non_churned = self.telco[self.telco['Churn'] == 0]
        churned = self.telco[self.telco['Churn'] == 1]
        churned.info()

        self.telco['Internet Type'].fillna('No Internet Type', inplace=True)
        self.telco['Offer'].fillna('Regular Plan', inplace=True)
        print(self.telco.isna().sum())

    def set_data_types(self) -> None:
        """Set appropriate data types for columns.
        
        Converts columns to their proper data types and maps categorical
        variables to numerical values where appropriate.
        """
        self.telco[['Customer ID', 'Zip Code']] = self.telco[['Customer ID', 'Zip Code']].astype(str)
        as_category = [
            'Churn Category', 'Churn Reason', 'City', 'Country',
            'Customer Status', 'Internet Type', 'Offer', 'Payment Method',
            'Quarter', 'State'
        ]
        self.telco[as_category] = self.telco[as_category].astype('category')
        self.telco['Gender'] = self.telco['Gender'].map({'Male': 1, 'Female': 0})
        print(self.telco.dtypes)

    def exploratory_data_analysis(self) -> None:
        """Perform initial exploratory data analysis.
        
        Creates subsets of data for churned and non-churned customers,
        calculates derived metrics, and prepares data for visualization.
        """
        self.churned = self.telco[self.telco['Churn'] == 1]
        self.non_churned = self.telco[self.telco['Churn'] == 0]

        print(self.churned['Tenure in Months'].describe())
        print(self.non_churned['Tenure in Months'].describe())
        print(self.churned['Satisfaction Score'].describe())
        print(self.non_churned['Satisfaction Score'].describe())

        # Calculate revenue per month
        self.telco['Revenue / Tenure in Months'] = (
            self.telco['Total Revenue'] / self.telco['Tenure in Months']
        )
        churn_by_revenue_month = self.telco.groupby('Churn')['Revenue / Tenure in Months'].median()

        # Create recent joiner flag
        self.telco['Recent Joiner'] = (self.telco['Tenure in Months'] < 7).astype('int')
        churn_by_recent_joiner = self.telco.groupby('Churn')['Recent Joiner'].value_counts()

        # Create age groups
        age_labels = ['18-24', '25-34', '35-44', '45-54', '+55']
        age_ranges = [0, 25, 35, 45, 55, self.telco['Age'].max()]
        self.telco['Age_levels'] = pd.cut(
            self.telco['Age'],
            bins=age_ranges,
            labels=age_labels
        )
        self.age_levels = self.telco['Age_levels']

        # Create label mappings
        self.telco['Churn_label'] = self.telco['Churn'].map({
            0: 'Active Users',
            1: 'Churned Users'
        })
        self.telco['Recent_Joiner_label'] = self.telco['Recent Joiner'].map({
            0: 'More than 6 months',
            1: 'Less 6 months'
        })

        # Define feature groups
        self._define_feature_groups()

    def _define_feature_groups(self) -> None:
        """Define groups of features for different types of analysis."""
        # Product features
        self.product_data = self.telco[[
            'Avg Monthly GB Download', 'Contract', 'Device Protection Plan',
            'Internet Type', 'Multiple Lines', 'Offer', 'Satisfaction Score',
            'Streaming Movies', 'Streaming Music', 'Streaming TV',
            'Unlimited Data', 'Churn'
        ]].copy()

        self.p_categorical_cols = [
            'Contract', 'Device Protection Plan', 'Internet Type',
            'Multiple Lines', 'Offer', 'Streaming Movies', 'Streaming Music',
            'Streaming TV', 'Unlimited Data'
        ]

        # Customer features
        self.customer_data = self.telco[[
            'Age', 'Age_levels', 'Gender', 'Number of Dependents',
            'Number of Referrals', 'Payment Method', 'Premium Tech Support',
            'Referred a Friend', 'Churn'
        ]].copy()

        self.c_categorical_cols = [
            'Age_levels', 'Gender', 'Payment Method',
            'Premium Tech Support', 'Referred a Friend'
        ]

        # Business features
        self.business_data = self.telco[[
            'CLTV', 'Monthly Charge', 'Tenure in Months', 'Total Charges',
            'Total Extra Data Charges', 'Total Revenue', 'Churn'
        ]].copy()

    def plot_churn_stacked_bar(
        self,
        crosstab_var: str,
        title: str,
        subtitle: str,
        xlabel: str
    ) -> None:
        """Create a stacked bar chart showing churn rates.
        
        Args:
            crosstab_var (str): Variable to analyze.
            title (str): Plot title.
            subtitle (str): Plot subtitle.
            xlabel (str): X-axis label.
        """
        g = pd.crosstab(
            self.telco[crosstab_var],
            self.telco['Churn'],
            normalize='index'
        ) * 100
        g = g[[0, 1]]

        ax = g.plot(
            kind='bar',
            stacked=True,
            color=self.colors,
            figsize=(8, 6),
            edgecolor='black'
        )

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%.1f%%',
                label_type='center',
                fontsize=10,
                color='white',
                fontweight='bold'
            )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.ylabel("Percentage of Users", fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(['Active Users', 'Churned Users'], title='User Status')
        plt.tight_layout()
        plt.show()

    def plot_churn_box_chart(
        self,
        ylabel: str,
        title: str,
        subtitle: str
    ) -> None:
        """Create a box plot comparing churned and active users.
        
        Args:
            ylabel (str): Y-axis label.
            title (str): Plot title.
            subtitle (str): Plot subtitle.
        """
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")

        g = sns.boxplot(
            data=self.telco,
            x='Churn_label',
            y=ylabel,
            palette=self.colors
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.xlabel('', fontsize=12)
        plt.xticks(rotation=0)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        subtitle: str,
        label_encode_cols: Optional[List[str]] = None,
        figsize: tuple = (10, 8)
    ) -> None:
        """Create a correlation heatmap.
        
        Args:
            data (pd.DataFrame): Data to analyze.
            title (str): Plot title.
            subtitle (str): Plot subtitle.
            label_encode_cols (Optional[List[str]]): Columns to label encode.
            figsize (tuple): Figure size.
        """
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

    def plot_kde_distribution(
        self,
        feature: str,
        title: str,
        subtitle: str,
        hue: str = 'Churn',
        label_map: Dict[int, str] = {0: 'Active Users', 1: 'Churned Users'},
        show_medians: bool = True,
        xlim: Optional[tuple] = None
    ) -> None:
        """Create a KDE plot showing distribution of a feature.
        
        Args:
            feature (str): Feature to plot.
            title (str): Plot title.
            subtitle (str): Plot subtitle.
            hue (str): Variable to color by.
            label_map (Dict[int, str]): Mapping for hue labels.
            show_medians (bool): Whether to show median lines.
            xlim (Optional[tuple]): X-axis limits.
        """
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
                plt.axvline(
                    median_val,
                    color=color,
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7
                )
                plt.text(
                    median_val,
                    plt.ylim()[1] * 0.03,
                    f'Median: {int(median_val)}',
                    color=color,
                    ha='center',
                    fontsize=9
                )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title='User Status', labels=['Churned Users', 'Active Users'])
        plt.tight_layout()
        plt.show()

    def plot_recent_vs_established_users(self) -> None:
        """Create a plot comparing recent and established users."""
        plt.figure(figsize=(8, 6))
        ax8 = sns.countplot(
            data=self.telco,
            x='Churn_label',
            hue='Recent_Joiner_label',
            palette=self.colors,
            edgecolor='black'
        )

        for p in ax8.patches:
            height = p.get_height()
            ax8.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height + 30),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='white'
            )

        plt.title(
            'Recent vs. Established Users by Churn Status',
            fontsize=14,
            fontweight='bold'
        )
        plt.suptitle(
            "Nearly half of churned users had left within the first 6 months of tenure",
            fontsize=10,
            color='gray'
        )
        plt.xlabel('User Churn Status', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(
            title='User Tenure',
            title_fontsize=11,
            fontsize=10,
            loc='upper right',
            labels=['More than 6 months', 'Less than 6 months']
        )
        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_revenue_per_month(self) -> None:
        """Create a plot showing revenue per month for different user groups."""
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
            ax.annotate(
                f'{height:.1f}',
                (p.get_x() + p.get_width() / 2., height + 1),
                ha='center',
                fontsize=11,
                fontweight='bold',
                color='black'
            )

        plt.title(
            'Revenue per Month: Active vs. Churned Users',
            fontsize=14,
            fontweight='bold'
        )
        plt.suptitle(
            "Revenue per month is higher among churned users, hinting at aggressive short-term monetization",
            fontsize=10,
            color='gray'
        )
        plt.xlabel('')
        plt.ylabel('Revenue / Month (€)', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_revenue_extra_data_charges(self) -> None:
        """Create a scatter plot of revenue vs extra data charges."""
        sns.scatterplot(
            data=self.telco,
            x='Total Revenue',
            y='Total Extra Data Charges',
            hue='Churn_label',
            palette=self.colors,
            edgecolor='black'
        )

        plt.title(
            'Revenue & Extra Data Charges: Active vs. Churned Users',
            fontsize=14,
            fontweight='bold'
        )
        plt.suptitle(
            "A few users that churned were charged for extra data usage while not generating much revenue",
            fontsize=10,
            color='gray'
        )
        plt.xlabel('Revenue / Month (€)')
        plt.ylabel('Total Extra Data Charges', fontsize=12)
        plt.tight_layout()
        plt.show()
