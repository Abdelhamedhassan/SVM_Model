
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler

class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.map_dict = {'Yes': 1, 'No': 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert X back to DataFrame if it's a NumPy array
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        return X.replace(self.map_dict)

class SexMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.map_dict = {'Male': 0, 'Female': 1}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert X back to DataFrame if it's a NumPy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=['Sex'])
        return X.replace(self.map_dict)
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.factor * IQR
            self.upper_bounds_[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        for col in X.columns:
            lower = self.lower_bounds_[col]
            upper = self.upper_bounds_[col]
            X[col] = np.clip(X[col], lower, upper)
        return X
    
class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)

        # Get feature names out from the column transformer
        self.feature_names_ = self._get_feature_names(X)
        print(self.feature_names_)

        return self

    def transform(self, X):
        # Transform the data
        X_t = self.preprocessor.transform(X)
        # Wrap it back into a DataFrame with correct column names
        return pd.DataFrame(X_t, columns=self.feature_names_, index=X.index)

    def _get_feature_names(self, X):
        output_features = []

        for name, transformer, cols in self.preprocessor.transformers_:
            if transformer == 'drop':
                continue
            elif transformer == 'passthrough':
                if isinstance(cols, slice):
                    passthrough_cols = X.columns[cols]
                else:
                    passthrough_cols = cols
                output_features.extend(passthrough_cols)
            else:
                # For Pipelines, get the last step
                if isinstance(transformer, Pipeline):
                    last_step = transformer.steps[-1][1]
                else:
                    last_step = transformer

                # Handle OneHotEncoder separately
                if hasattr(last_step, 'get_feature_names_out'):
                    # Use the original column names
                    names = last_step.get_feature_names_out(cols)
                    output_features.extend(names)
                else:
                    if isinstance(cols, slice):
                        cols = X.columns[cols]
                    output_features.extend(cols)

        return [str(name).replace('__', '_') for name in output_features]
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold=0.01,
        use_variance=True,
        use_mutual_info=True,
        use_fisher=True,
        use_chi2=True,
        top_k_mi=20,
        top_k_fisher=5,
        top_k_chi2=20
    ):
        self.threshold = threshold
        self.use_variance = use_variance
        self.use_mutual_info = use_mutual_info
        self.use_fisher = use_fisher
        self.use_chi2 = use_chi2
        self.top_k_mi = top_k_mi
        self.top_k_fisher = top_k_fisher
        self.top_k_chi2 = top_k_chi2

    def fit(self, X, y=None):
        # Convert X to DataFrame for consistent column tracking
        if isinstance(X, np.ndarray):
            self.original_features = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.original_features)
        else:
            self.original_features = X.columns.tolist()

        self.X_ = X
        self.y_ = y

        self.selected_features_var = []
        self.selected_features_mi = []
        self.selected_features_fisher = []
        self.selected_features_chi = []

        if self.use_variance:
            print("\n===== Variance Threshold Feature Selection =====")
            selector = VarianceThreshold(threshold=self.threshold)
            selector.fit(X)
            self.variances_ = selector.variances_
            self.selected_features_var = X.columns[selector.get_support()].tolist()

            feature_variances = pd.Series(selector.variances_, index=X.columns)


        if self.use_mutual_info:
            print("\n===== Mutual Information Feature Selection =====")
            mi_features = [
                "Sex", "GeneralHealth", "LastCheckupTime", "PhysicalActivities", "RemovedTeeth",
                "HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer",
                "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes",
                "DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating",
                "DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands",
                "AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver",
                "HighRiskLastYear", "CovidPos", "TetanusLast10Tdap", "AgeCategory"
            ]
            mi_features = [f for f in mi_features if f in X.columns]
            X_mi = X[mi_features]
            mi = mutual_info_classif(X_mi, y, random_state=42)
            mi_df = pd.DataFrame({'Feature': X_mi.columns, 'MI_Score': mi})
            selected_mi_df = mi_df.sort_values('MI_Score', ascending=False).head(self.top_k_mi)
            self.selected_features_mi = selected_mi_df['Feature'].tolist()

            # print("\nTop 15 features by Mutual Information score:")
            # print(selected_mi_df)

        if self.use_fisher:
            print("\n===== Fisher Score (ANOVA F-value) Feature Selection =====")
            features_fisher = [
                'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours',
                'HeightInMeters', 'WeightInKilograms', 'BMI', 'TetanusLast10Tdap'
            ]
            features_fisher = [f for f in features_fisher if f in X.columns]
            f_scores, _ = f_classif(X[features_fisher], y)
            fisher_series = pd.Series(f_scores, index=features_fisher)
            self.selected_features_fisher = fisher_series.sort_values(ascending=False).head(self.top_k_fisher).index.tolist()

            # print("\nTop 5 features by Fisher Score:")
            # print(fisher_series.sort_values(ascending=False).head(5))

        if self.use_chi2:
            print("\n===== Chi-squared Feature Selection =====")
            features_chi2 = [
                'Sex', 'GeneralHealth', 'PhysicalActivities', 'LastCheckupTime',
                'RemovedTeeth', 'HadAngina', 'HadStroke', 'HadAsthma',
                'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease',
                'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
                'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
                'DifficultyErrands', 'ChestScan', 'AgeCategory', 'AlcoholDrinkers',
                'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear'
            ] + [col for col in X.columns if col.startswith('State_')
                 or col.startswith('SmokerStatus_')
                 or col.startswith('ECigaretteUsage_')
                 or col.startswith('RaceEthnicityCategory_')]

            features_chi2 = [f for f in features_chi2 if f in X.columns]
            X_cat = X[features_chi2]
            X_cat_scaled = MinMaxScaler().fit_transform(X_cat)

            chi_vals, _ = chi2(X_cat_scaled, y)
            chi_series = pd.Series(chi_vals, index=features_chi2)
            self.selected_features_chi = chi_series.sort_values(ascending=False).head(self.top_k_chi2).index.tolist()

            # print("\nTop 10 features by Chi-squared test:")
            # print(chi_series.sort_values(ascending=False).head(10))

            # Combine all selected features
            self.final_features_ = list(
                (set(self.selected_features_var) & set(self.selected_features_mi)) |
                set(self.selected_features_fisher) |
                set(self.selected_features_chi)
            )



        print(f"\nTotal selected features (combined): {len(self.final_features_)}")
        print(f"\nselected features (combined): {self.final_features_}")

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.original_features)
        return X[self.final_features_].values
