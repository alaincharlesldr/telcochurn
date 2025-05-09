# Import libraries
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

# Define Class & Initialize
class ChurnPredictionModel:
    def __init__(self, data, target_column):
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
        self.product_features_w_score = ['Avg Monthly GB Download', 'Contract', 'Device Protection Plan', 'Internet Type',
                                        'Multiple Lines', 'Offer', 'Satisfaction Score', 'Streaming Movies',
                                        'Streaming Music', 'Streaming TV', 'Unlimited Data']
        self.product_features_wo_score = [f for f in self.product_features_w_score if f != 'Satisfaction Score']

# Data preprocessing

    def prepare_data(self):
        # Separate features and target
        self.y = self.data[self.target_column]
        self.X1 = self.data[self.product_features_w_score].copy()
        self.X2 = self.data[self.product_features_wo_score].copy()
        self._encode_categorical_variables()

    def _encode_categorical_variables(self):
        # Define variables to encode
        variables_to_encode = ['Contract', 'Internet Type', 'Offer']
        encoded_variables = pd.DataFrame(index=self.data.index)

        # Encode each categorical variable
        for var in variables_to_encode:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            transformed = encoder.fit_transform(self.data[[var]])
            encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([var]), index=self.data.index)
            self.encoders[var] = encoder
            encoded_variables = pd.concat([encoded_variables, encoded_df], axis=1)

        # Update X1 and X2 with encoded variables
        self.X1 = pd.concat([self.X1.drop(columns=variables_to_encode), encoded_variables], axis=1).astype('float64')
        self.X2 = pd.concat([self.X2.drop(columns=variables_to_encode), encoded_variables], axis=1).astype('float64')


# Split the data

    def split_data(self):
        # Split data into training and test sets
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in split.split(self.X1, self.y):
            self.X1_train, self.X1_test = self.X1.iloc[train_idx], self.X1.iloc[test_idx]
            self.X2_train, self.X2_test = self.X2.iloc[train_idx], self.X2.iloc[test_idx]
            self.y_train, self.y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        
        # Scale data
        self.scale_data()

    def scale_data(self):
        
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
    
    def train_and_evaluate_models(self):
        """Main orchestrator"""
        self._train_evaluate("With Satisfaction Score", self.X1_train, self.X1_test, self.results_with_score, self.churn_risk_with_score)
        self._train_evaluate("Without Satisfaction Score", self.X2_train, self.X2_test, self.results_without_score, self.churn_risk_without_score)
    
    
    def _train_evaluate(self, name_suffix, X_train, X_test, results_list, churn_risk_list):
        """Reusable evaluator"""
        
        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
    
    
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}
    
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
        for name, model in models.items():
            # CV Scores
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=kf, scoring='roc_auc')
            results_list.append(pd.DataFrame({'Model': name, 'CV Accuracy Scores': cv_scores}))
            print(f"[{name_suffix}] {name} - Mean ROC-AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
            # Train & predict
            model.fit(X_train, self.y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            churn_risk_list.append(pd.DataFrame({'Model': name, 'Churn Probability': y_proba}))
    
            # Store Logistic Regression models for feature importance
            if name == "Logistic Regression":
                if name_suffix == "With Satisfaction Score":
                    self.models_with_score[name] = model
                elif name_suffix == "Without Satisfaction Score":
                    self.models_without_score[name] = model

    
    def _plot_comparisons(self):
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
    
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}
    
        # Plot churn probability
    
        self._plot_churn_probability_distribution(
                self.churn_risk_with_score, 
                list(models.keys()),
                model_colors, 
                "With Satisfaction Score", 
                subtitle="With the Satisfaction Score feature, all models display a similar churn probability distribution curve")
            
        self._plot_churn_probability_distribution(
                self.churn_risk_without_score, 
                list(models.keys()),
                model_colors, 
                "Without Satisfaction Score", 
                subtitle="Without the Satisfaction Score feature, KNN and Decision Tree models show more irregular churn probability distributions")  
        
        # Plot ROC-AUC distribution
    
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
                subtitle = "Even without the Satisfaction Score feature, the Logistic Regression model returns the highest median ROC-AUC score. \n KNN has the highest variance and returns a ROC-AUC scores close to the Logistic Regression model")

# Plotting functions
    
    # Churn probability distribution
    def _plot_churn_probability_distribution(self, churn_risk, model_names, model_colors, title_suffix, subtitle):
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
        
        churn_risk_df = pd.concat(churn_risk, ignore_index=True)
        churn_risk_df['Model'] = pd.Categorical(churn_risk_df['Model'], categories=list(models.keys()), ordered=True)

        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=churn_risk_df, 
                    x='Churn Probability', 
                    hue='Model', 
                    hue_order= list(reversed(list(model_names))), 
                    fill=True, 
                    linewidth=2.5, 
                    alpha=0.5, 
                    palette=model_colors)
        
        
        plt.axvline(0.2, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label="Low risk threshold")
        plt.axvline(0.6, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label="High risk threshold")
        plt.title(f"Churn Probability Distribution {title_suffix}", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Churn Probability", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.legend(title="Model", labels=model_names, fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.show()

    # ROC-AUC distribution
    def _plot_roc_auc_distribution(self, results, model_names, model_colors, title_suffix, subtitle):
        results_df = pd.concat(results)

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=results_df, 
                    x="Model", 
                    y="CV Accuracy Scores", 
                    palette=model_colors)
        
        plt.ylabel("Cross-Validation ROC-AUC", fontsize=12)
        plt.title(f"Comparing ROC-AUCs Distribution {title_suffix}", fontsize=16, fontweight="bold")
        plt.suptitle(subtitle, fontsize=10, color='gray')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def evaluate_logistic_regression(self):
        model_with_score = LogisticRegression(max_iter=1000)
        model_wo_score = LogisticRegression(max_iter=1000)

        model_with_score.fit(self.X1_train, self.y_train)
        model_wo_score.fit(self.X2_train, self.y_train)

        y1_proba = model_with_score.predict_proba(self.X1_test)[:, 1]
        y2_proba = model_wo_score.predict_proba(self.X2_test)[:, 1]

        self._plot_roc_curve(y1_proba, y2_proba)
        self._print_classification_report(model_with_score, model_wo_score)

# Logistic Regression Evaluation - roc curve, confusion matrix & classification report
    def _plot_roc_curve(self, y1_proba, y2_proba):
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

    def _print_classification_report(self, model_with_score, model_wo_score):
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

    def feature_importance(self):
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



    def _plot_feature_importance(self, importance_df, title_suffix):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='Coefficient', y='Feature', palette='coolwarm')
        plt.title(f'Top Feature Importances (Logistic Regression, {title_suffix})', fontsize=14, fontweight='bold')
        plt.axvline(0, color='gray', linestyle='--')
        plt.tight_layout()
        plt.show()

# Adding & evaluating Random Forest (without Satisfaction Score)

    def random_forest_analysis(self):
        rf = RandomForestClassifier(n_estimators=25, random_state=42)
        rf.fit(self.X2_train, self.y_train)
        rf_y_proba = rf.predict_proba(self.X2_test)[:, 1]
        rf_roc_auc = roc_auc_score(self.y_test, rf_y_proba)
        print(f"Random Forest ROC-AUC: {rf_roc_auc:.4f}")
        
        importances = pd.Series(data=rf.feature_importances_, index=self.X2_train.columns)
        importances_sorted = importances.sort_values()
        importances_sorted.plot(kind='barh', color='lightgreen')
        plt.title('Features Importances')
        plt.show()

        result = permutation_importance(rf, self.X2_test, self.y_test, n_repeats=10, random_state=42)
        perm_importances = pd.Series(result.importances_mean, index=self.X2_test.columns)
        perm_importances_sorted = perm_importances.sort_values()
        perm_importances_sorted.plot(kind='barh', color='lightblue')
        plt.title('Permutation Feature Importances')
        plt.show()

        self.random_forest_model = rf
        self._evaluate_random_forest()
        
    
    def _evaluate_random_forest(self):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": self.random_forest_model
        }
        
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}

        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        rf_cv = cross_val_score(self.random_forest_model, self.X2_train, self.y_train, cv=kf)
        self.results_without_score.append(pd.DataFrame({"Model": ["Random Forest"] * len(rf_cv), "CV Accuracy Scores": rf_cv}))
        
        # ROC-AUCs Distribution, RF & Other models
        self._plot_roc_auc_distribution(
            self.results_without_score,
            list(models.keys()),
            model_colors,
            "With Random Forest", 
            subtitle="Random Forest shows significant ROC-AUC score improvements compared to Decision Tree, but still not surpassing Logistic Regression")

        # Churn Probability Distribution, RF & Other models
        self.models_without_score["Random Forest"] = self.random_forest_model
        
        rf_y_proba = self.random_forest_model.predict_proba(self.X2_test)[:, 1]
        self.churn_risk_without_score.append(pd.DataFrame({'Model': 'Random Forest', 'Churn Probability': rf_y_proba}))

        self._plot_churn_probability_distribution(
            self.churn_risk_without_score, 
            list(models.keys()),
            model_colors, 
            "With Random Forest", 
            subtitle="Random Forest displays the smoothest churn probability density distribution, in-between all three other models")



# Hyperparameter tuning
    def hyperparameter_tuning(self):
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": self.random_forest_model
        }
        
        set2_palette = sns.color_palette("Set2", len(models))
        model_colors = {name: color for name, color in zip(models.keys(), set2_palette)}
                
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2)
        grid_search.fit(self.X2_train, self.y_train)
        print("Best Parameters:", grid_search.best_params_)
        print("Best ROC-AUC:", grid_search.best_score_)

        rf_tuned = RandomForestClassifier(**grid_search.best_params_)
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        rf_tuned_cv = cross_val_score(rf_tuned, self.X2_train, self.y_train, cv=kf)
        self.results_without_score.append(pd.DataFrame({"Model": ["Random Forest"] * len(rf_tuned_cv), "CV Accuracy Scores": rf_tuned_cv}))

        self._plot_roc_auc_distribution(
            self.results_without_score, 
            list(models.keys()),
            model_colors, 
            "With Tuned Random Forest",
            subtitle="Fine-tuned Random Forest model is still not performing as well as the Logistic Regression model"
        )

    # Scoring Churn Risk
    def churn_risk_scoring(self):
        # Copy raw columns
        X_full = self.data[self.product_features_w_score].copy()
        
        # Encode categorical columns
        encoded_variables_full = pd.DataFrame(index=self.data.index)
    
        for var, encoder in self.encoders.items():
            transformed = encoder.transform(self.data[[var]])
            encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([var]), index=self.data.index)
            encoded_variables_full = pd.concat([encoded_variables_full, encoded_df], axis=1)
    
        # Combine features
        X_full = pd.concat([X_full.drop(columns=self.encoders.keys()), encoded_variables_full], axis=1).astype('float64')

        # Scaling data
        X_full = pd.DataFrame(
            self.scaler_X1.transform(X_full),
            columns=X_full.columns,
            index=X_full.index
        )
        
        # Predict
        self.data["Churn Probability Score"] = self.models_with_score["Logistic Regression"].predict_proba(X_full)[:, 1]
    
        # Plot
        self._plot_churn_score_relationship()

    def _plot_churn_score_relationship(self):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.data, 
            x='Churn Probability Score', 
            y='Churn Score', 
            alpha=0.4, 
            edgecolor=None)
        sns.regplot(
            data=self.data, 
            x='Churn Probability Score', 
            y='Churn Score', 
            scatter=False, 
            color='red')
        plt.title("Relationship between Model Churn Probability and initial Churn Score", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Churn Probability", fontsize=12)
        plt.ylabel("Initial Churn Score", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage
    
# Load & import dataset
from utils.huggingface_loader import load_huggingface_dataset
telco = load_huggingface_dataset("aai510-group1/telco-customer-churn")
    
# Instantiate ChurnPredictionModel
model = ChurnPredictionModel(telco, 'Churn')
    
# Preparing our explanatory variables & our target variable
model.prepare_data()
    
# Splitting/scale our data & training our model
model.split_data()
model.train_and_evaluate_models()

# Plot comparisons
model._plot_comparisons()
    
# Get ROC Curves & classification reports
model.evaluate_logistic_regression()
    
# Visualize feature importance
model.feature_importance()
    
# Random Forest Analysis
model.random_forest_analysis()
    
# Hyperparameter Tuning
model.hyperparameter_tuning()
    
# Churn Risk Scoring
model.churn_risk_scoring()
