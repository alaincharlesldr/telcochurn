"""Exploratory Data Analysis Package.

This package provides tools for analyzing and visualizing customer churn data
in the telecommunications industry.

Main Features:
- Data loading and preprocessing
- Exploratory data analysis
- Churn rate visualization
- Correlation analysis
- Revenue analysis
- Customer behavior analysis

Example usage:

    from eda.data_analyzer import DataAnalyzer

    analyzer = DataAnalyzer(dataset_path='path/to/dataset')
    analyzer.exploratory_data_analysis()
    analyzer.plot_churn_stacked_bar()
    analyzer.plot_churn_box_chart()
    analyzer.plot_correlation_heatmap()
    analyzer.plot_kde_distribution()
    analyzer.plot_revenue_per_month()
    analyzer.plot_revenue_extra_data_charges()
    analyzer.plot_recent_vs_established_users()
"""

from eda.data_analyzer import DataAnalyzer

__all__ = ['DataAnalyzer']
