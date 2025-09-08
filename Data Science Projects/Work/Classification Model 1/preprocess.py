import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("C:\python\IDS ASSIGNMENT\Codes\Classification Model 1\parkinson+s+disease+classification\pd_speech_features\pd_speech_features.csv")
X = df.drop(columns=["class"])
y = df["class"]

# 1. Remove Zero-Variance Features
variance_threshold = VarianceThreshold(threshold=0.01)
X_var = variance_threshold.fit_transform(X)
var_columns = X.columns[variance_threshold.get_support()]  # keep names
X_var = pd.DataFrame(X_var, columns=var_columns)

# 2. Remove Highly Correlated Features
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

X_corr = remove_correlated_features(X_var, threshold=0.95)

# 3. Univariate Feature Selection (ANOVA F-value)
selector = SelectKBest(score_func=f_classif, k=100)
X_kbest = selector.fit_transform(X_corr, y)
kbest_columns = X_corr.columns[selector.get_support()]
X_kbest = pd.DataFrame(X_kbest, columns=kbest_columns)

# 4. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_kbest)
X_scaled_df = pd.DataFrame(X_scaled, columns=kbest_columns)

# Split data for RFE
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# 5. Recursive Feature Elimination (RFE)
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=50, step=10)
X_final = rfe.fit_transform(X_train, y_train)
rfe_columns = X_train.columns[rfe.get_support()]
X_final_df = pd.DataFrame(X_final, columns=rfe_columns)

# Add target and save
X_final_df["target"] = y_train.values
X_final_df.to_csv("reduced_dataset.csv", index=False)

# Diagnostics
print(f"Original features: {X.shape[1]}")
print(f"After variance threshold: {len(var_columns)}")
print(f"After correlation removal: {X_corr.shape[1]}")
print(f"After SelectKBest: {X_kbest.shape[1]}")
print(f"Final selected features (after RFE): {X_final.shape[1]}")
print("Selected feature names:", rfe_columns.tolist())
