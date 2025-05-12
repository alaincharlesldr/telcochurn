"""Telco Customer Churn Analysis - Streamlit Application

This is the main Streamlit application for the Telco Customer Churn Analysis project.
It provides an interactive interface for exploring customer churn data, viewing
analysis results, and making predictions.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set matplotlib backend to Agg before importing pyplot
import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import base64

# Import project modules
try:
    from utils.huggingface_loader import load_huggingface_dataset
    from models.churn_model import ChurnPredictionModel
    from eda.data_analyzer import DataAnalyzer
except ImportError as e:
    st.error(f"""
    Error importing required modules: {str(e)}
    
    Please ensure you are running the app from the project root directory:
    ```bash
    cd /path/to/TelcoChurn
    streamlit run streamlit_app/app.py
    ```
    """)
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ChurnModel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


def load_data():
    """Load and cache the dataset."""
    @st.cache_data
    def _load_data():
        return load_huggingface_dataset("aai510-group1/telco-customer-churn")
    return _load_data()


def initialize_models():
    """Initialize and cache the analysis and prediction models."""
    @st.cache_resource
    def _initialize_models():
        data = load_data()
        analyzer = DataAnalyzer(data)
        # Initialize analyzer data
        analyzer.exploratory_data_analysis()
        
        # Initialize and train the model
        model = ChurnPredictionModel(data, target_column='Churn')
        model.prepare_data()
        model.split_data()
        model.train_and_evaluate_models()
        
        return analyzer, model
    return _initialize_models()


def main():
    """Main application function."""
    st.title("📊 ChurnModel")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "",
        ["Overview", "Data Analysis", "Modeling", "Simulator", "About"]
    )
    
    # Load data and models
    try:
        data = load_data()
        analyzer, model = initialize_models()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    if page == "Overview":
        show_dashboard(data, analyzer)
    elif page == "Data Analysis":
        show_eda_dashboard(data, analyzer)
    elif page == "Modeling":
        show_model_comparison(model)
    elif page == "Simulator":
        show_churn_prediction(data, model)
    else:  # About page
        show_about()


def show_dashboard(data: pd.DataFrame, analyzer: DataAnalyzer):
    """Display the main dashboard with key metrics and visualizations."""
    st.header("🔸 Overview")
    
    # Create a copy of the data with explicit labels
    plot_data = data.copy()
    plot_data['Status'] = plot_data['Churn'].map({0: 'Active', 1: 'Churned'})
    
    # Key metrics in a grid layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🧾 Total Customers",
            f"{len(data):,}"
        )
        st.metric(
            "📊 Total Variables",
            "52"
        )
    
    with col2:
        churn_rate = (data['Churn'].mean() * 100)
        st.metric(
            "🔁 Ratio Churned",
            f"{churn_rate:.1f}%"
        )
        st.metric(
            "😃 Avg. Satisfaction",
            "3.2/5"
        )
    
    with col3:
        avg_revenue = data['Total Revenue'].mean()
        st.metric(
            "💸 Avg Revenue per user",
            f"${avg_revenue:,.2f}"
        )
        active_tenure = data[data['Churn'] == 0]['Tenure in Months'].mean()
        st.metric(
            "🕒 Avg. Tenure (Active)",
            f"{active_tenure:.0f} months"
        )
    
    with col4:
        st.metric(
            "⌛ Avg. Tenure (Churned)",
            "10 months"
        )
        st.metric(
            "📈 Tenure Ratio",
            f"{active_tenure/10:.1f}x"
        )
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn Distribution Pie Chart
        status_counts = plot_data['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Customer Churn Distribution',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Satisfaction Score Distribution
        fig = px.box(
            plot_data,
            x='Status',
            y='Satisfaction Score',
            title='Satisfaction Score Distribution by Churn Status',
            color='Status',
            color_discrete_map={'Active': '#2ecc71', 'Churned': '#e74c3c'}
        )
        fig.update_layout(
            xaxis_title="Customer Status",
            yaxis_title="Satisfaction Score",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenure Distribution
        fig = px.histogram(
            plot_data,
            x='Tenure in Months',
            color='Status',
            title='Tenure Distribution by Churn Status',
            color_discrete_map={'Active': '#2ecc71', 'Churned': '#e74c3c'},
            nbins=30,
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(
            xaxis_title="Tenure (Months)",
            yaxis_title="Number of Customers",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue Distribution
        fig = px.box(
            plot_data,
            x='Status',
            y='Total Revenue',
            title='Revenue Distribution by Churn Status',
            color='Status',
            color_discrete_map={'Active': '#2ecc71', 'Churned': '#e74c3c'}
        )
        fig.update_layout(
            xaxis_title="Customer Status",
            yaxis_title="Total Revenue ($)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


def show_eda_dashboard(data: pd.DataFrame, analyzer: DataAnalyzer):
    """Display EDA dashboard with comprehensive visualizations."""
    st.header("📊 Exploratory Data Analysis Dashboard")
    
    # Ensure analyzer data is initialized
    if not hasattr(analyzer, 'product_data') or analyzer.product_data is None:
        analyzer.exploratory_data_analysis()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📘 Customer Features", "📘 Product Features", "📘 Business Features"])
    
    with tab1:
        st.subheader("Customer Features Analysis")
        st.markdown("""
        Analysis of customer demographics and behavior:
        - Gender and age distribution
        - Payment methods and preferences
        - Referral activity
        - Customer satisfaction patterns
        """)
        
        # Customer Demographics
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender Analysis
            with plt.rc_context({'backend': 'Agg'}):
                analyzer.plot_churn_stacked_bar(
                    crosstab_var='Gender',
                    title="Churn Rate by Gender",
                    subtitle="Churn is at similar levels across Male and Female usergroups",
                    xlabel="Gender \n Female: O, Male: 1"
                )
                st.pyplot(plt.gcf())
                plt.close()
        
        with col2:
            # Friend Referral Analysis
            with plt.rc_context({'backend': 'Agg'}):
                analyzer.plot_churn_stacked_bar(
                    crosstab_var='Referred a Friend',
                    title="Churn Rate by Referral Activity",
                    subtitle="Users that Referred a Friend are more sticky",
                    xlabel="Referred a Friend? \n No: 0, Yes: 1"
                )
                st.pyplot(plt.gcf())
                plt.close()
        
        # Payment Method Analysis
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_churn_stacked_bar(
                crosstab_var='Payment Method',
                title="Churn Rate by Payment Method",
                subtitle="Credit Card is associated with lower churn levels",
                xlabel="Payment Method"
            )
            st.pyplot(plt.gcf())
            plt.close()
        
        # Age Analysis
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_churn_stacked_bar(
                crosstab_var='Age_levels',
                title="Churn Rate by Age",
                subtitle="Older age groups tend to have higher churn",
                xlabel="Age"
            )
            st.pyplot(plt.gcf())
            plt.close()
        
        # Customer Features Heatmap
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_correlation_heatmap(
                data=analyzer.customer_data,
                title='Customer Features Correlation with Churn',
                subtitle='Satisfaction Score is the strongest negative churn predictor — Contract type also stands out',
                label_encode_cols=analyzer.c_categorical_cols
            )
            st.pyplot(plt.gcf())
            plt.close()
    
    with tab2:
        st.subheader("Product Features Analysis")
        st.markdown("""
        Analysis of service and product usage:
        - Contract types and durations
        - Internet service types
        - Additional services and features
        - Data usage patterns
        """)
        
        # Contract Type Analysis
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_churn_stacked_bar(
                crosstab_var='Contract',
                title="Churn Rate by Contract Type",
                subtitle="Longer contracts are associated with lower churn rates",
                xlabel="Contract Type"
            )
            st.pyplot(plt.gcf())
            plt.close()
        
        # Offer Type Analysis
        with plt.rc_context({'backend': 'Agg'}):
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=data,
                y='Offer',
                x='Satisfaction Score',
                palette='viridis'
            )
            plt.title("Average Satisfaction Score by Offer Type", fontsize=14, fontweight='bold')
            plt.xlabel("Average Satisfaction Score", fontsize=12)
            plt.ylabel("Offer Type", fontsize=12)
            st.pyplot(plt.gcf())
            plt.close()
        
        # Internet Type Analysis
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_churn_stacked_bar(
                crosstab_var='Internet Type',
                title="Churn Rate by Internet Type",
                subtitle="Fiber Optic users show higher churn rates",
                xlabel="Internet Type"
            )
            st.pyplot(plt.gcf())
            plt.close()
        
        # Data Usage Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            with plt.rc_context({'backend': 'Agg'}):
                plt.figure(figsize=(8, 6))
                sns.boxplot(
                    data=data,
                    x='Churn',
                    y='Avg Monthly GB Download',
                    palette=['#2ecc71', '#e74c3c']
                )
                plt.title("Data Usage for Active vs. Churned Users", fontsize=14, fontweight='bold')
                plt.xlabel("User Status", fontsize=12)
                plt.ylabel("Monthly GB Download", fontsize=12)
                st.pyplot(plt.gcf())
                plt.close()
        
        with col2:
            with plt.rc_context({'backend': 'Agg'}):
                plt.figure(figsize=(8, 6))
                sns.boxplot(
                    data=data,
                    x='Churn',
                    y='Satisfaction Score',
                    palette=['#2ecc71', '#e74c3c']
                )
                plt.title("Churn vs. Satisfaction Score", fontsize=14, fontweight='bold')
                plt.xlabel("User Status", fontsize=12)
                plt.ylabel("Satisfaction Score", fontsize=12)
                st.pyplot(plt.gcf())
                plt.close()
        
        # Product Features Heatmap
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_correlation_heatmap(
                data=analyzer.product_data,
                title='Product Features Correlation with Churn',
                subtitle='Satisfaction Score is strongly negatively correlated with churn — Contract type also stands out',
                label_encode_cols=analyzer.p_categorical_cols
            )
            st.pyplot(plt.gcf())
            plt.close()
    
    with tab3:
        st.subheader("Business Features Analysis")
        st.markdown("""
        Analysis of business metrics and performance:
        - Revenue patterns and trends
        - Customer lifetime value
        - Tenure and retention
        - Extra charges and usage
        """)
        
        # Revenue Distribution
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_kde_distribution(
                feature='Total Revenue',
                title="Revenue Distribution across Active & Churned groups",
                subtitle="Every churned user represents revenue left on the table",
                xlim=(0, None)
            )
            st.pyplot(plt.gcf())
            plt.close()
        
        # Customer Lifetime Value Distribution
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_kde_distribution(
                feature='CLTV',
                title="Customer Lifetime Value Distribution across Active & Churned groups",
                subtitle="Churned users have significantly lower (c. 8-9% less) lifetime value than retained ones",
                xlim=(0, None)
            )
            st.pyplot(plt.gcf())
            plt.close()
        
        # Tenure Distribution
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_kde_distribution(
                feature='Tenure in Months',
                title="Tenure in Months Distribution by Churn",
                subtitle="Churn is happening in the first 6-12 months; Active users stay 4x longer",
                xlim=(0, None)
            )
            st.pyplot(plt.gcf())
            plt.close()
        
        # Recent vs Established Users
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_recent_vs_established_users()
            st.pyplot(plt.gcf())
            plt.close()
        
        # Revenue per Month
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_revenue_per_month()
            st.pyplot(plt.gcf())
            plt.close()
        
        # Revenue vs Extra Data Charges
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_revenue_extra_data_charges()
            st.pyplot(plt.gcf())
            plt.close()
        
        # Business Features Heatmap
        with plt.rc_context({'backend': 'Agg'}):
            analyzer.plot_correlation_heatmap(
                data=analyzer.business_data,
                title='Business Features Correlation with Churn',
                subtitle='Churn is negatively correlated with Customer Lifetime Value and Revenues',
                label_encode_cols=None
            )
            st.pyplot(plt.gcf())
            plt.close()


def show_model_comparison(model: ChurnPredictionModel):
    """Display model comparison plots from demo_model.py."""
    st.header("🤖 Model Comparison Dashboard")
    
    # Reset model's data structures to prevent accumulation
    model.results_without_score = [df for df in model.results_without_score if df['Model'].iloc[0] != 'Random Forest']
    model.churn_risk_without_score = [df for df in model.churn_risk_without_score if df['Model'].iloc[0] != 'Random Forest']
    if 'Random Forest' in model.models_without_score:
        del model.models_without_score['Random Forest']
    
    # Define all possible model names and create a consistent color palette
    all_models = [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ]
    
    # Create a consistent color palette for all models
    model_colors = dict(zip(
        all_models,
        sns.color_palette("Set2", len(all_models))
    ))
    
    # Plot comparisons
    st.subheader("Model Performance Comparison")
    with plt.rc_context({'backend': 'Agg'}):
        plt.figure(figsize=(12, 8))
        # Plot churn probability distributions
        model._plot_churn_probability_distribution(
            model.churn_risk_with_score, 
            all_models[:3],  # Exclude Random Forest for initial plots
            model_colors, 
            "With Satisfaction Score", 
            subtitle="With the Satisfaction Score feature, all models display a similar churn probability distribution curve"
        )
        st.pyplot(plt.gcf())
        plt.close()
        
        plt.figure(figsize=(12, 8))
        model._plot_churn_probability_distribution(
            model.churn_risk_without_score, 
            all_models[:3],  # Exclude Random Forest for initial plots
            model_colors, 
            "Without Satisfaction Score", 
            subtitle="Without the Satisfaction Score feature, KNN and Decision Tree models show more irregular churn probability distributions"
        )
        st.pyplot(plt.gcf())
        plt.close()
        
        plt.figure(figsize=(12, 8))
        # Plot ROC-AUC distributions
        model._plot_roc_auc_distribution(
            model.results_with_score, 
            all_models[:3],  # Exclude Random Forest for initial plots
            model_colors, 
            "With Satisfaction Score", 
            subtitle="With the Satisfaction Score feature, the Logistic Regression model outperforms all other models"
        )
        st.pyplot(plt.gcf())
        plt.close()
        
        plt.figure(figsize=(12, 8))
        model._plot_roc_auc_distribution(
            model.results_without_score,
            all_models[:3],  # Exclude Random Forest for initial plots
            model_colors, 
            "Without Satisfaction Score", 
            subtitle="Even without the Satisfaction Score feature, the Logistic Regression model returns the highest median ROC-AUC score. \n KNN has the highest variance and returns a ROC-AUC scores close to the Logistic Regression model"
        )
        st.pyplot(plt.gcf())
        plt.close()
    
    # Evaluate Logistic Regression
    st.subheader("Logistic Regression Evaluation")
    with plt.rc_context({'backend': 'Agg'}):
        plt.figure(figsize=(10, 6))
        model.evaluate_logistic_regression()
        st.pyplot(plt.gcf())
        plt.close()
    
    # Feature Importance
    st.subheader("Feature Importance Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        with plt.rc_context({'backend': 'Agg'}):
            plt.figure(figsize=(10, 6))
            model.feature_importance()
            st.pyplot(plt.gcf())
            plt.close()
    
    with col2:
        with plt.rc_context({'backend': 'Agg'}):
            plt.figure(figsize=(10, 6))
            # Get feature importance from Logistic Regression
            lr_model = model.models_with_score["Logistic Regression"]
            feature_importance = pd.DataFrame({
                'Feature': model.X1_train.columns,
                'Importance': np.abs(lr_model.coef_[0])
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=True)
            
            # Plot horizontal bar chart
            plt.barh(feature_importance['Feature'], feature_importance['Importance'])
            plt.title('Logistic Regression Feature Importance')
            plt.xlabel('Absolute Coefficient Value')
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
    
    # Random Forest Analysis
    st.subheader("Random Forest Analysis")
    with plt.rc_context({'backend': 'Agg'}):
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=25, random_state=42)
        rf.fit(model.X2_train, model.y_train)
        rf_y_proba = rf.predict_proba(model.X2_test)[:, 1]
        rf_roc_auc = roc_auc_score(model.y_test, rf_y_proba)
        
        # Plot impurity-based feature importance
        plt.figure(figsize=(10, 6))
        importances = pd.Series(
            data=rf.feature_importances_,
            index=model.X2_train.columns
        )
        importances_sorted = importances.sort_values()
        importances_sorted.plot(kind='barh', color='lightgreen')
        plt.title('Features Importances')
        st.pyplot(plt.gcf())
        plt.close()
        
        # Plot permutation-based feature importance
        plt.figure(figsize=(10, 6))
        result = permutation_importance(
            rf,
            model.X2_test,
            model.y_test,
            n_repeats=10,
            random_state=42
        )
        perm_importances = pd.Series(
            result.importances_mean,
            index=model.X2_test.columns
        )
        perm_importances_sorted = perm_importances.sort_values()
        perm_importances_sorted.plot(kind='barh', color='lightblue')
        plt.title('Permutation Feature Importances')
        st.pyplot(plt.gcf())
        plt.close()
        
        # Store Random Forest model and predictions
        model.models_without_score["Random Forest"] = rf
        
        # Calculate cross-validation scores
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        rf_cv = cross_val_score(rf, model.X2_train, model.y_train, cv=kf)
        model.results_without_score.append(pd.DataFrame({
            "Model": ["Random Forest"] * len(rf_cv),
            "CV Accuracy Scores": rf_cv
        }))
        
        # Add Random Forest predictions
        model.churn_risk_without_score.append(pd.DataFrame({
            'Model': 'Random Forest',
            'Churn Probability': rf_y_proba
        }))
        
        # Plot ROC-AUC distribution with Random Forest
        plt.figure(figsize=(10, 6))
        model._plot_roc_auc_distribution(
            model.results_without_score,
            all_models,  # Include Random Forest
            model_colors,
            "With Random Forest", 
            subtitle="Random Forest shows significant ROC-AUC score improvements compared to Decision Tree, but still not surpassing Logistic Regression"
        )
        st.pyplot(plt.gcf())
        plt.close()
        
        # Plot churn probability distribution with Random Forest
        plt.figure(figsize=(10, 6))
        model._plot_churn_probability_distribution(
            model.churn_risk_without_score, 
            all_models,  # Include Random Forest
            model_colors, 
            "With Random Forest", 
            subtitle="Random Forest displays the smoothest churn probability density distribution, in-between all three other models"
        )
        st.pyplot(plt.gcf())
        plt.close()
    
    # Hyperparameter Tuning Results
    st.subheader("Hyperparameter Tuning Results")
    with plt.rc_context({'backend': 'Agg'}):
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5]
        }
        
        # Perform grid search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(model.X2_train, model.y_train)
        
        # Evaluate tuned model
        rf_tuned = RandomForestClassifier(**grid_search.best_params_)
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        rf_tuned_cv = cross_val_score(rf_tuned, model.X2_train, model.y_train, cv=kf)
        
        # Update results with tuned model
        model.results_without_score = [df for df in model.results_without_score if df['Model'].iloc[0] != 'Random Forest']
        model.results_without_score.append(pd.DataFrame({
            "Model": ["Random Forest"] * len(rf_tuned_cv),
            "CV Accuracy Scores": rf_tuned_cv
        }))
        
        # Plot ROC-AUC distribution with tuned Random Forest
        plt.figure(figsize=(10, 6))
        model._plot_roc_auc_distribution(
            model.results_without_score, 
            all_models,
            model_colors, 
            "With Tuned Random Forest",
            subtitle="Fine-tuned Random Forest model is still not performing as well as the Logistic Regression model"
        )
        st.pyplot(plt.gcf())
        plt.close()


def show_churn_prediction(data: pd.DataFrame, model: ChurnPredictionModel):
    """Display the churn prediction interface."""
    st.header("🎯 Churn Prediction")
    
    # Prediction mode selection
    mode = st.radio(
        "Select Prediction Mode",
        ["Single Customer", "Batch Prediction"]
    )
    
    if mode == "Single Customer":
        show_single_prediction(model)
    else:
        show_batch_prediction(model)


def show_about():
    """Display information about the project."""
    # Stack and Dataset in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Stack
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Matplotlib, Seaborn, Plotly
        - **Machine Learning**: Scikit-learn
        - **Frontend**: Streamlit for interactive web interface
        - **Version Control**: Git
        """)
    
    with col2:
        st.markdown("""
        #### Dataset
        The dataset is sourced from Hugging Face and contains customer data from a telecommunications company.
        
        #### Version
        This is version 1 of this project. Next versions will include other classification models (such as XGBoost) and dimensionality reduction.
        """)
    
    # About me section with round profile picture
    st.markdown("---")  # Add a separator
    st.markdown("#### About me")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Add your profile picture with round styling
        st.markdown("""
        <style>
        .profile-img {
            border-radius: 50%;
            width: 100%;
            max-width: 300px;
            margin: 0 auto;
            display: block;
            object-fit: cover;
            aspect-ratio: 1;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Use the image from assets directory with HTML/CSS for rounded styling
        st.markdown(
            f'<img src="data:image/jpeg;base64,{base64.b64encode(open("streamlit_app/assets/profile.jpg", "rb").read()).decode()}" class="profile-img">',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("""
        For the past 4 years, I've been working across business operations, sales & project management - first as a startup founder and then as a Strategic Project Manager at Homa. Prior to that, I was an M&A and Private Equity Analyst. I'm now back into leveraging my analytical skills, but this time applied to Data Science! 
        
        You can find more details about my work on my [GitHub](https://github.com/alaincharlesldr/telcochurn) and read my article about this project on [Medium](https://medium.com/@alaincharlesldr/a-60-uplift-in-churn-recall-the-case-for-prioritizing-customer-surveys-9cb473e9685e).
        
        And if you'd like to get in touch ☕️, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/alain-charles-lauriano-do-rego-7001b089/) or drop me an [email](mailto:alaincharlesldr@gmail.com).
        """)


# Helper functions for data analysis
def show_demographics_analysis(data: pd.DataFrame):
    """Display demographic analysis visualizations."""
    st.subheader("Customer Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(
            data,
            x='Age',
            color='Churn',
            title='Age Distribution by Churn Status',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        fig = px.pie(
            data,
            names='Gender',
            color='Churn',
            title='Gender Distribution by Churn Status',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_service_analysis(data: pd.DataFrame):
    """Display service usage analysis visualizations."""
    st.subheader("Service Usage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Internet type distribution
        fig = px.bar(
            data.groupby(['Internet Type', 'Churn']).size().reset_index(name='count'),
            x='Internet Type',
            y='count',
            color='Churn',
            title='Internet Type Distribution by Churn Status',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Service usage correlation
        fig = px.scatter(
            data,
            x='Avg Monthly GB Download',
            y='Total Revenue',
            color='Churn',
            title='Service Usage vs Revenue by Churn Status',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_financial_analysis(data: pd.DataFrame):
    """Display financial analysis visualizations."""
    st.subheader("Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue distribution
        fig = px.box(
            data,
            x='Churn',
            y='Total Revenue',
            title='Revenue Distribution by Churn Status',
            color='Churn',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payment method analysis
        fig = px.bar(
            data.groupby(['Payment Method', 'Churn']).size().reset_index(name='count'),
            x='Payment Method',
            y='count',
            color='Churn',
            title='Payment Method Distribution by Churn Status',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_churn_patterns(data: pd.DataFrame):
    """Display churn pattern analysis visualizations."""
    st.subheader("Churn Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn reasons
        fig = px.bar(
            data.groupby('Churn Reason').size().reset_index(name='count'),
            x='Churn Reason',
            y='count',
            title='Top Churn Reasons',
            color='count',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Satisfaction impact
        fig = px.scatter(
            data,
            x='Satisfaction Score',
            y='Total Revenue',
            color='Churn',
            title='Satisfaction Impact on Churn',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_single_prediction(model: ChurnPredictionModel):
    """Display interface for single customer prediction."""
    st.subheader("Predict Churn for a Single Customer")
    
    # Create input form
    with st.form("prediction_form"):
        # First row: Age and Revenue
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=45, help="Age is displayed for reference only and does not affect the prediction")
        with col2:
            total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, max_value=10000.0, value=1000.0, help="Revenue is displayed for reference only and does not affect the prediction")
        
        # Second row: Internet and Contract
        col1, col2 = st.columns(2)
        with col1:
            internet_type = st.selectbox(
                "Internet Type",
                ["Fiber Optic", "Cable", "DSL", "No Internet Service"]
            )
        with col2:
            contract = st.selectbox(
                "Contract",
                ["Month-to-Month", "One Year", "Two Year"]
            )
        
        # Third row: Satisfaction Score and Offer
        col1, col2 = st.columns(2)
        with col1:
            satisfaction_score = st.slider(
                "Satisfaction Score",
                min_value=1,
                max_value=5,
                value=3
            )
        with col2:
            offer = st.selectbox(
                "Offer",
                ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
            )
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            # Create input data with features in the exact order expected by the model
            input_data = pd.DataFrame({
                'Avg Monthly GB Download': [50.0],  # Median value
                'Contract': [contract],
                'Device Protection Plan': [0],
                'Internet Type': [internet_type],
                'Multiple Lines': [0],
                'Offer': [offer],
                'Satisfaction Score': [satisfaction_score],
                'Streaming Movies': [0],
                'Streaming Music': [0],
                'Streaming TV': [0],
                'Unlimited Data': [0]
            })
            
            # Make prediction
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)
                
                # Display results
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Get the probability for the predicted class
                    pred_prob = probability[0][1] if prediction[0] == 1 else probability[0][0]
                    # Set color based on prediction
                    color = "red" if prediction[0] == 1 else "green"
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {color}20;'>
                        <h3 style='color: {color};'>Churn Prediction</h3>
                        <h2 style='color: {color};'>{'Likely to Churn' if prediction[0] == 1 else 'Likely to Stay'}</h2>
                        <p style='color: {color}; font-size: 1.2em;'>{pred_prob*100:.1f}% probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Top 5 Key Factors")
                    # Display top 5 factors influencing the prediction
                    factors = model.get_feature_importance(input_data)
                    # Sort and get top 5
                    top_factors = dict(sorted(factors.items(), key=lambda x: x[1], reverse=True)[:5])
                    for factor, importance in top_factors.items():
                        st.markdown(f"- {factor}: {importance:.1%}")
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def show_batch_prediction(model: ChurnPredictionModel):
    """Display interface for batch prediction."""
    st.subheader("Batch Churn Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            data = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = [
                'Internet Type', 'Contract', 'Offer', 'Satisfaction Score'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Add default values for additional model features
            data['Avg Monthly GB Download'] = 50.0  # Median value
            data['Device Protection Plan'] = 0
            data['Multiple Lines'] = 0
            data['Streaming Movies'] = 0
            data['Streaming Music'] = 0
            data['Streaming TV'] = 0
            data['Unlimited Data'] = 0
            
            # Make predictions
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)
            
            # Add predictions to data
            data['Churn Prediction'] = predictions
            data['Churn Probability'] = probabilities[:, 1]
            
            # Display results
            st.subheader("Prediction Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Customers",
                    len(data)
                )
            with col2:
                churn_count = predictions.sum()
                st.metric(
                    "Predicted Churns",
                    churn_count,
                    f"{churn_count/len(data)*100:.1f}%"
                )
            with col3:
                st.metric(
                    "Avg. Churn Probability",
                    f"{probabilities[:, 1].mean()*100:.1f}%"
                )
            
            # Display detailed results
            st.dataframe(data)
            
            # Download results
            csv = data.to_csv(index=False)
            st.download_button(
                "Download Predictions",
                csv,
                "churn_predictions.csv",
                "text/csv",
                key='download-csv'
            )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main() 