import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
file_path = r"reduced_dataset.csv"
data = pd.read_csv(file_path)

# Clean columns: Remove extra spaces
data.columns = data.columns.str.strip().str.replace(' ', '')

# Data cleaning: Drop duplicates
data.drop_duplicates(inplace=True)

# Handle missing values
num_cols = data.select_dtypes(include=np.number).columns.tolist()  # Numeric columns
cat_cols = data.select_dtypes(include=['object']).columns.tolist()  # Categorical columns

# Fill missing values for numeric columns with median
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Fill missing values for categorical columns with mode
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Identify the target column (assuming the last column is the target)
target_col = data.columns[-1]  # Replace with your target column if it's not the last one
X = data.drop(columns=[target_col])  # Features
y = data[target_col]  # Target variable

feature_sizes = [2, 4, 8, 16, 32, 45, X.shape[1]]



# Function to evaluate models for growing feature subsets
def evaluate_by_feature_count(X, y, feature_sizes, model_type='knn'):
    results = []
    for i, size in enumerate(feature_sizes, start=1):
        # Select the first 'size' features
        selected_features = X.columns[:size]
        X_subset = X[selected_features]
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.3, random_state=42
        )
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Initialize model
        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'nb':
            model = GaussianNB()
        # Train the model and make predictions
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Calculate accuracy and confusion matrix
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        # Store the results
        results.append({
            'step': i,
            'num_features': size,
            'accuracy': acc,
            'confusion_matrix': cm,
            'features_used': selected_features.tolist()
        })
     # Save confusion matrix plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_type.upper()} (Step {i}: {size} features)')
        plt.savefig(f'{model_type}_step{i}.png')
        plt.close()

    return results


# Run evaluation for KNN and Naive Bayes
print("Evaluating models with growing feature subsets...")
knn_results = evaluate_by_feature_count(X, y, feature_sizes, 'knn')
nb_results = evaluate_by_feature_count(X, y, feature_sizes, 'nb')

# Create a comparison table
comparison = pd.DataFrame({
    'Step': [res['step'] for res in knn_results],
    'Num_Features': [res['num_features'] for res in knn_results],
    'KNN_Accuracy': [res['accuracy'] for res in knn_results],
    'NB_Accuracy': [res['accuracy'] for res in nb_results],
})

# Display the performance comparison
print("\nPerformance Comparison:")
print(comparison.to_string(index=False))

# Plot performance comparison
plt.figure(figsize=(10, 5))
plt.plot(comparison['Num_Features'], comparison['KNN_Accuracy'], marker='o', label='KNN')
plt.plot(comparison['Num_Features'], comparison['NB_Accuracy'], marker='o', label='Naive Bayes')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Model Performance vs Feature Count')
plt.legend()
plt.grid(True)
plt.savefig('feature_count_vs_accuracy.png')
plt.close()

# ---- Bivariate Plotting ----

# Bivariate analysis: Scatterplot and Boxplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='numPulses', y='mean_MFCC_2nd_coef', hue='target', data=data)
plt.title('Scatterplot: numPulses vs mean_MFCC_2nd_coef by Target')
plt.savefig('bivariate_scatter_numPulses_vs_mean_MFCC_2nd_coef.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='maxIntensity', data=data)
plt.title('Boxplot: maxIntensity by Target')
plt.savefig('bivariate_boxplot_maxIntensity_by_target.png')
plt.close()

# Bivariate analysis: Correlation heatmap (top 10 features)
top_features = X.columns[:10]  # Select top 10 features for correlation analysis
correlation_matrix = data[top_features].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Top 10 Features')
plt.savefig('correlation_heatmap_top_10_features.png')
plt.close()

print("\nAnalysis complete! Generated:")
print("- Confusion matrix plots (2 models × feature stages)")
print("- Performance comparison plot")
print("- Bivariate analysis plots (scatter, box, and correlation heatmap)")
