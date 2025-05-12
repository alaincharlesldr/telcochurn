"""Churn Prediction Model for Telco Customer Analysis.

This module implements a churn prediction model using various machine learning
algorithms to analyze customer churn patterns in telecommunications data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class ChurnPredictionModel:
    """A class to implement and evaluate churn prediction models.
    
    This class provides functionality for data preprocessing, model training,
    evaluation, and visualization of churn prediction results using various
    machine learning algorithms.
    
    Attributes:
        data (pd.DataFrame): The input dataset containing customer information.
        target_column (str): The name of the target column (Churn status).
        X1 (pd.DataFrame): Explanatory features including satisfaction score.
        X2 (pd.DataFrame): Explanatory features excluding satisfaction score.
        y (pd.Series): Target variable.
        encoders (dict): Dictionary storing encoders for categorical variables.
        models_with_score (dict): Dictionary storing models trained with the Satisfaction Score feature.
        models_without_score (dict): Dictionary storing models trained without the Satisfaction Score feature.
        results_with_score (list): List storing results for models with the Satisfaction Score feature.
        results_without_score (list): List storing results for models without the Satisfaction Score feature.
        churn_risk_with_score (list): List storing churn risk predictions with the Satisfaction Score feature.
        churn_risk_without_score (list): List storing churn risk predictions without the Satisfaction Score feature.
    """

    def __init__(self, data: pd.DataFrame, target_column: str) -> None:
        """Initialize the ChurnPredictionModel.
        
        Args:
            data (pd.DataFrame): The input dataset containing customer information.
            target_column (str): The name of the target column (Churn status).
        """
        self.data = data
        self.target_column = target_column
        self.X1 = None
        self.X2 = None
        self.y = None
        self.encoders = {}
        self.models_with_score = {}
        self.models_without_score = {}
        self.results_with_score = []
        self.results_without_score = []
        self.churn_risk_with_score = []
        self.churn_risk_without_score = []
        self.product_features_w_score = [
            'Avg Monthly GB Download', 'Contract', 'Device Protection Plan',
            'Internet Type', 'Multiple Lines', 'Offer', 'Satisfaction Score',
            'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Unlimited Data'
        ]
        self.product_features_wo_score = [
            f for f in self.product_features_w_score 
            if f != 'Satisfaction Score'
        ]

# Data preprocessing

    def prepare_data(self) -> None:
        """Prepare the data for model training.
        
        This method separates features and target variables, and encodes categorical
        variables using one-hot encoding.
        """
        self.y = self.data[self.target_column]
        self.X1 = self.data[self.product_features_w_score].copy()
        self.X2 = self.data[self.product_features_wo_score].copy()
        self._encode_categorical_variables()

    def _encode_categorical_variables(self) -> None:
        """Encode categorical variables using one-hot encoding.
        
        This private method handles the encoding of categorical variables and updates
        the feature matrices X1 and X2 with the encoded variables.
        """
        variables_to_encode = ['Contract', 'Internet Type', 'Offer']
        encoded_variables = pd.DataFrame(index=self.data.index)

        for var in variables_to_encode:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            transformed = encoder.fit_transform(self.data[[var]])
            encoded_df = pd.DataFrame(
                transformed,
                columns=encoder.get_feature_names_out([var]),
                index=self.data.index
            )
            self.encoders[var] = encoder
            encoded_variables = pd.concat([encoded_variables, encoded_df], axis=1)

        self.X1 = pd.concat(
            [self.X1.drop(columns=variables_to_encode), encoded_variables],
            axis=1
        ).astype('float64')
        
        self.X2 = pd.concat(
            [self.X2.drop(columns=variables_to_encode), encoded_variables],
            axis=1
        ).astype('float64')

    def split_data(self) -> None:
        """Split the data into training and test sets.
        
        This method performs stratified splitting of the data and scales the features
        using StandardScaler.
        """
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in split.split(self.X1, self.y):
            self.X1_train, self.X1_test = self.X1.iloc[train_idx], self.X1.iloc[test_idx]
            self.X2_train, self.X2_test = self.X2.iloc[train_idx], self.X2.iloc[test_idx]
            self.y_train, self.y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        
        self.scale_data()

    def scale_data(self) -> None:
        """Scale the features using StandardScaler.
        
        This method scales both training and test sets for features with and without
        satisfaction score.
        """
        self.scaler_X1 = StandardScaler()
        self.scaler_X2 = StandardScaler()
    
        self.X1_train = pd.DataFrame(
            self.scaler_X1.fit_transform(self.X1_train),
            columns=self.X1.columns,
            index=self.X1_train.index
        )
        
        self.X1_test = pd.DataFrame(
            self.scaler_X1.transform(self.X1_test),
            columns=self.X1.columns,
            index=self.X1_test.index
        )
    
        self.X2_train = pd.DataFrame(
            self.scaler_X2.fit_transform(self.X2_train),
            columns=self.X2.columns,
            index=self.X2_train.index
        )
        
        self.X2_test = pd.DataFrame(
            self.scaler_X2.transform(self.X2_test),
            columns=self.X2.columns,
            index=self.X2_test.index
        )

# Train & evaluate models
    
    def train_and_evaluate_models(self) -> None:
        """Train and evaluate models with and without satisfaction score.
        
        This method orchestrates the training and evaluation of models using both
        feature sets (with and without satisfaction score).
        """
        self._train_evaluate(
            "With Satisfaction Score",
            self.X1_train,
            self.X1_test,
            self.results_with_score,
            self.churn_risk_with_score
        )
        self._train_evaluate(
            "Without Satisfaction Score",
            self.X2_train,
            self.X2_test,
            self.results_without_score,
            self.churn_risk_without_score
        )
    
    def _train_evaluate(self, name_suffix: str, X_train: pd.DataFrame, X_test: pd.DataFrame, results_list: list, churn_risk_list: list) -> None:
        """Train and evaluate models for a specific feature set.
        
        Args:
            name_suffix (str): Suffix to identify the feature set being used.
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            results_list (list): List to store cross-validation results.
            churn_risk_list (list): List to store churn risk predictions.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
    
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}
    
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
        for name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=kf, scoring='roc_auc')
            results_list.append(pd.DataFrame({'Model': name, 'CV Accuracy Scores': cv_scores}))
            print(f"[{name_suffix}] {name} - Mean ROC-AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
            # Train and predict
            model.fit(X_train, self.y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            churn_risk_list.append(pd.DataFrame({'Model': name, 'Churn Probability': y_proba}))
    
            # Store Logistic Regression models for feature importance
            if name == "Logistic Regression":
                if name_suffix == "With Satisfaction Score":
                    self.models_with_score[name] = model
                elif name_suffix == "Without Satisfaction Score":
                    self.models_without_score[name] = model

    def _plot_comparisons(self) -> None:
        """Plot comparison visualizations for model performance.
        
        This method generates plots comparing model performance with and without
        satisfaction score, including churn probability distribution (KDEplot) and ROC-AUC scores distribution (boxplot).
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
    
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}
    
        # Plot churn probability distributions
        self._plot_churn_probability_distribution(
            self.churn_risk_with_score, 
            list(models.keys()),
            model_colors, 
            "With Satisfaction Score", 
            subtitle="With the Satisfaction Score feature, all models display a similar churn probability distribution curve"
        )
            
        self._plot_churn_probability_distribution(
            self.churn_risk_without_score, 
            list(models.keys()),
            model_colors, 
            "Without Satisfaction Score", 
            subtitle="Without the Satisfaction Score feature, KNN and Decision Tree models show more irregular churn probability distributions"
        )  
        
        # Plot ROC-AUC distributions
        self._plot_roc_auc_distribution(
            self.results_with_score, 
            list(models.keys()),
            model_colors, 
            "With Satisfaction Score", 
            subtitle="With the Satisfaction Score feature, the Logistic Regression model outperforms all other models"
        )
            
        self._plot_roc_auc_distribution(
            self.results_without_score,
            list(models.keys()),
            model_colors, 
            "Without Satisfaction Score", 
            subtitle="Even without the Satisfaction Score feature, the Logistic Regression model returns the highest median ROC-AUC score. \n KNN has the highest variance and returns a ROC-AUC scores close to the Logistic Regression model"
        )

# Plotting functions
    
    def _plot_churn_probability_distribution(
        self,
        churn_risk: list,
        model_names: list,
        model_colors: dict,
        title_suffix: str,
        subtitle: str
    ) -> None:
        """Plot the distribution of churn probabilities for different models.
        
        Args:
            churn_risk (list): List of DataFrames containing churn probabilities.
            model_names (list): List of model names to plot.
            model_colors (dict): Dictionary mapping model names to colors.
            title_suffix (str): Suffix for the plot title.
            subtitle (str): Subtitle for the plot.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
        
        churn_risk_df = pd.concat(churn_risk, ignore_index=True)
        churn_risk_df['Model'] = pd.Categorical(
            churn_risk_df['Model'],
            categories=list(models.keys()),
            ordered=True
        )

        plt.figure(figsize=(8, 6))
        sns.kdeplot(
            data=churn_risk_df, 
            x='Churn Probability', 
            hue='Model', 
            hue_order=list(reversed(model_names)), 
            fill=True, 
            linewidth=2.5, 
            alpha=0.5, 
            palette=model_colors
        )
        
        plt.axvline(0.2, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label="Low risk threshold")
        plt.axvline(0.6, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label="High risk threshold")
        plt.title(f"Churn Probability Distribution {title_suffix}", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Churn Probability", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.legend(title="Model", labels=model_names, fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.show()

    def _plot_roc_auc_distribution(
        self,
        results: list,
        model_names: list,
        model_colors: dict,
        title_suffix: str,
        subtitle: str
    ) -> None:
        """Plot the distribution of ROC-AUC scores for different models.
        
        Args:
            results (list): List of DataFrames containing ROC-AUC scores.
            model_names (list): List of model names to plot.
            model_colors (dict): Dictionary mapping model names to colors.
            title_suffix (str): Suffix for the plot title.
            subtitle (str): Subtitle for the plot.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }

        results_df = pd.concat(results)

        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=results_df, 
            x="Model", 
            y="CV Accuracy Scores", 
            palette=model_colors
        )
        
        plt.ylabel("Cross-Validation ROC-AUC", fontsize=12)
        plt.title(f"Comparing ROC-AUCs Distribution {title_suffix}", fontsize=16, fontweight="bold")
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Evaluating Logistic Regression models - initialisation
    
    def evaluate_logistic_regression(self) -> None:
        """Evaluate Logistic Regression models with and without satisfaction score.
        
        This method trains and evaluates Logistic Regression models using both feature sets,
        generates ROC curves, and prints classification reports.
        """
        model_with_score = LogisticRegression(max_iter=1000)
        model_wo_score = LogisticRegression(max_iter=1000)

        model_with_score.fit(self.X1_train, self.y_train)
        model_wo_score.fit(self.X2_train, self.y_train)

        y1_proba = model_with_score.predict_proba(self.X1_test)[:, 1]
        y2_proba = model_wo_score.predict_proba(self.X2_test)[:, 1]

        self._plot_roc_curve(y1_proba, y2_proba)
        self._print_classification_report(model_with_score, model_wo_score)

    def _plot_roc_curve(self, y1_proba: np.ndarray, y2_proba: np.ndarray) -> None:
        """Plot ROC curves for models with and without satisfaction score.
        
        Args:
            y1_proba (np.ndarray): Predicted probabilities for model with satisfaction score.
            y2_proba (np.ndarray): Predicted probabilities for model without satisfaction score.
        """
        fpr1, tpr1, _ = roc_curve(self.y_test, y1_proba)
        fpr2, tpr2, _ = roc_curve(self.y_test, y2_proba)

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

    def _print_classification_report(
        self,
        model_with_score: LogisticRegression,
        model_wo_score: LogisticRegression
    ) -> None:
        """Print classification reports for both models.
        
        Args:
            model_with_score (LogisticRegression): Model trained with satisfaction score.
            model_wo_score (LogisticRegression): Model trained without satisfaction score.
        """
        y1_pred = model_with_score.predict(self.X1_test)
        y2_pred = model_wo_score.predict(self.X2_test)

        print("📊 Confusion Matrix: With Satisfaction Score")
        print(confusion_matrix(self.y_test, y1_pred))
        print("🧾 Classification Report: With Satisfaction Score")
        print(classification_report(self.y_test, y1_pred))

        print("📊 Confusion Matrix: Without Satisfaction Score")
        print(confusion_matrix(self.y_test, y2_pred))
        print("🧾 Classification Report: Without Satisfaction Score")
        print(classification_report(self.y_test, y2_pred))

# Feature importance

    def feature_importance(self) -> None:
        """Analyze and plot feature importance for both models.
        
        This method calculates and visualizes feature importance for Logistic Regression
        models with and without satisfaction score.
        """
        # With Satisfaction Score
        importance1_df = pd.DataFrame({
            'Feature': self.X1_train.columns,
            'Coefficient': self.models_with_score["Logistic Regression"].coef_[0]
        })
        importance1_df['Abs_Coefficient'] = importance1_df['Coefficient'].abs()
        importance1_df = importance1_df.sort_values(by='Abs_Coefficient', ascending=False)
        self._plot_feature_importance(importance1_df, "With Satisfaction Score")
    
        # Without Satisfaction Score
        importance2_df = pd.DataFrame({
            'Feature': self.X2_train.columns,
            'Coefficient': self.models_without_score["Logistic Regression"].coef_[0]
        })
        importance2_df['Abs_Coefficient'] = importance2_df['Coefficient'].abs()
        importance2_df = importance2_df.sort_values(by='Abs_Coefficient', ascending=False)
        self._plot_feature_importance(importance2_df, "Without Satisfaction Score")

    def _plot_feature_importance(self, importance_df: pd.DataFrame, title_suffix: str) -> None:
        """Plot feature importance for a model.
        
        Args:
            importance_df (pd.DataFrame): DataFrame containing feature importance data.
            title_suffix (str): Suffix for the plot title.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=importance_df.head(10),
            x='Coefficient',
            y='Feature',
            palette='coolwarm'
        )
        plt.title(
            f'Top Feature Importances (Logistic Regression, {title_suffix})',
            fontsize=14,
            fontweight='bold'
        )
        plt.axvline(0, color='gray', linestyle='--')
        plt.tight_layout()
        plt.show()

# Adding & evaluating Random Forest (without Satisfaction Score)

    def random_forest_analysis(self) -> None:
        """Perform Random Forest analysis without satisfaction score.
        
        This method trains a Random Forest model, evaluates its performance,
        and analyzes feature importance using both impurity-based and permutation
        importance methods.
        """
        rf = RandomForestClassifier(n_estimators=25, random_state=42)
        rf.fit(self.X2_train, self.y_train)
        rf_y_proba = rf.predict_proba(self.X2_test)[:, 1]
        rf_roc_auc = roc_auc_score(self.y_test, rf_y_proba)
        print(f"Random Forest ROC-AUC: {rf_roc_auc:.4f}")
        
        # Plot impurity-based feature importance
        importances = pd.Series(
            data=rf.feature_importances_,
            index=self.X2_train.columns
        )
        importances_sorted = importances.sort_values()
        importances_sorted.plot(kind='barh', color='lightgreen')
        plt.title('Features Importances')
        plt.show()

        # Plot permutation-based feature importance
        result = permutation_importance(
            rf,
            self.X2_test,
            self.y_test,
            n_repeats=10,
            random_state=42
        )
        perm_importances = pd.Series(
            result.importances_mean,
            index=self.X2_test.columns
        )
        perm_importances_sorted = perm_importances.sort_values()
        perm_importances_sorted.plot(kind='barh', color='lightblue')
        plt.title('Permutation Feature Importances')
        plt.show()

        self.random_forest_model = rf
        self._evaluate_random_forest()
    
    def _evaluate_random_forest(self) -> None:
        """Evaluate Random Forest model against other models.
        
        This method compares the Random Forest model's performance with other models previously analyzed
        using ROC-AUC scores and churn probability distributions.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": self.random_forest_model
        }
        
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}

        # Calculate cross-validation scores
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        rf_cv = cross_val_score(self.random_forest_model, self.X2_train, self.y_train, cv=kf)
        self.results_without_score.append(pd.DataFrame({
            "Model": ["Random Forest"] * len(rf_cv),
            "CV Accuracy Scores": rf_cv
        }))
        
        # Plot ROC-AUC distribution
        self._plot_roc_auc_distribution(
            self.results_without_score,
            list(models.keys()),
            model_colors,
            "With Random Forest", 
            subtitle="Random Forest shows significant ROC-AUC score improvements compared to Decision Tree, but still not surpassing Logistic Regression"
        )

        # Store Random Forest model and predictions
        self.models_without_score["Random Forest"] = self.random_forest_model
        rf_y_proba = self.random_forest_model.predict_proba(self.X2_test)[:, 1]
        self.churn_risk_without_score.append(pd.DataFrame({
            'Model': 'Random Forest',
            'Churn Probability': rf_y_proba
        }))

        # Plot churn probability distribution
        self._plot_churn_probability_distribution(
            self.churn_risk_without_score, 
            list(models.keys()),
            model_colors, 
            "With Random Forest", 
            subtitle="Random Forest displays the smoothest churn probability density distribution, in-between all three other models"
        )

    def hyperparameter_tuning(self) -> None:
        """Perform hyperparameter tuning for Random Forest model.
        
        This method uses GridSearchCV to find optimal hyperparameters for the
        Random Forest model and evaluates its performance with the tuned parameters.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": self.random_forest_model
        }
        
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}
                
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
        grid_search.fit(self.X2_train, self.y_train)
        print("Best Parameters:", grid_search.best_params_)
        print("Best ROC-AUC:", grid_search.best_score_)

        # Evaluate tuned model
        rf_tuned = RandomForestClassifier(**grid_search.best_params_)
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        rf_tuned_cv = cross_val_score(rf_tuned, self.X2_train, self.y_train, cv=kf)
        self.results_without_score.append(pd.DataFrame({
            "Model": ["Random Forest"] * len(rf_tuned_cv),
            "CV Accuracy Scores": rf_tuned_cv
        }))

        # Plot ROC-AUC distribution
        self._plot_roc_auc_distribution(
            self.results_without_score, 
            list(models.keys()),
            model_colors, 
            "With Tuned Random Forest",
            subtitle="Fine-tuned Random Forest model is still not performing as well as the Logistic Regression model"
        )

    def churn_risk_scoring(self) -> None:
        """Calculate and visualize churn risk scores for all customers.
        
        This method calculates churn probabilities for all customers using the
        Logistic Regression model with satisfaction score and visualizes the
        relationship between predicted probabilities and the "Churn Score" feature in initial dataset.
        """
        # Copy raw columns
        X_full = self.data[self.product_features_w_score].copy()
        
        # Encode categorical columns
        encoded_variables_full = pd.DataFrame(index=self.data.index)
    
        for var, encoder in self.encoders.items():
            transformed = encoder.transform(self.data[[var]])
            encoded_df = pd.DataFrame(
                transformed,
                columns=encoder.get_feature_names_out([var]),
                index=self.data.index
            )
            encoded_variables_full = pd.concat([encoded_variables_full, encoded_df], axis=1)
    
        # Combine features
        X_full = pd.concat(
            [X_full.drop(columns=self.encoders.keys()), encoded_variables_full],
            axis=1
        ).astype('float64')

        # Scale data
        X_full = pd.DataFrame(
            self.scaler_X1.transform(X_full),
            columns=X_full.columns,
            index=X_full.index
        )
        
        # Predict
        self.data["Churn Probability Score"] = self.models_with_score["Logistic Regression"].predict_proba(X_full)[:, 1]
    
        # Plot
        self._plot_churn_score_relationship()

    def _plot_churn_score_relationship(self) -> None:
        """Plot the relationship between predicted churn probability and the "Churn Score" feature.
        
        This method creates a scatter plot with a regression line to visualize
        the correlation between our model's churn predictions and the "Churn Score" feature in initial dataset.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.data, 
            x='Churn Probability Score', 
            y='Churn Score', 
            alpha=0.4, 
            edgecolor=None
        )
        sns.regplot(
            data=self.data, 
            x='Churn Probability Score', 
            y='Churn Score', 
            scatter=False, 
            color='red'
        )
        plt.title(
            "Relationship between Model Churn Probability and initial Churn Score",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Predicted Churn Probability", fontsize=12)
        plt.ylabel("Initial Churn Score", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def main() -> None:
    """Main function to demonstrate the usage of ChurnPredictionModel.
    
    This function loads the dataset, initializes the model, and runs through
    the complete analysis pipeline.
    """
    # Load dataset
    from utils.huggingface_loader import load_huggingface_dataset
    telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")
    
    # Initialize model
    model = ChurnPredictionModel(telco, 'Churn')
    
    # Prepare data
    model.prepare_data()
    
    # Split and scale data
    model.split_data()
    
    # Train and evaluate models
    model.train_and_evaluate_models()

    # Plot comparisons
    model._plot_comparisons()
    
    # Evaluate Logistic Regression
    model.evaluate_logistic_regression()
    
    # Analyze feature importance
    model.feature_importance()
    
    # Perform Random Forest analysis
    model.random_forest_analysis()
    
# Hyperparameter Tuning
model.hyperparameter_tuning()
    
# Churn Risk Scoring
model.churn_risk_scoring()
