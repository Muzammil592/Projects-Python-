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
file_path = "D:/IDS ASSIGNMENT/Datasets/htru2/HTRU_2.csv"  # Update this if needed
data = pd.read_csv(file_path)

# Rename class column for easier access
data.rename(columns={'class {0,1}': 'target'}, inplace=True)

# Define feature groups
all_features = data.columns[:-1].tolist()  # All columns except target
feature_sets = [
    (all_features[:2], "First 2 Features"),
    (all_features[:4], "First 4 Features"),
    (all_features, "All 8 Features")
]


# Bivariate Scatter Plot for first two features
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=data,
    x=all_features[0],  # Mean of the integrated profile
    y=all_features[1],  # Standard deviation of the integrated profile
    hue='target',
    palette='coolwarm',
    alpha=0.7
)
plt.title('Bivariate Plot: Mean vs. Std of Integrated Profile (HTRU_2)')
plt.xlabel(all_features[0])
plt.ylabel(all_features[1])
plt.legend(title='Target')
plt.savefig('bivariate_mean_std.png')
plt.close()

# Optional: Pairplot for first 4 features
sns.pairplot(
    data[all_features[:4] + ['target']],
    hue='target',
    palette='coolwarm',
    plot_kws={'alpha': 0.6}
)
plt.suptitle('Pairplot of First 4 Features', y=1.02)
plt.savefig('pairplot_first4.png')
plt.close()

# Function to evaluate models
def evaluate_models(data, feature_sets, model_type='knn'):
    results = []
    for i, (features, label) in enumerate(feature_sets, start=1):
        X = data[features]
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=5) if model_type == 'knn' else GaussianNB()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'step': i,
            'num_features': len(features),
            'accuracy': acc,
            'confusion_matrix': cm,
            'feature_label': label
        })

        # Save confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_type.upper()} (Step {i}: {label})')
        plt.savefig(f'{model_type}_step{i}.png')
        plt.close()

    return results

# Run evaluations
print("Evaluating models at all 3 feature levels...")
knn_results = evaluate_models(data, feature_sets, 'knn')
nb_results = evaluate_models(data, feature_sets, 'nb')

# Create performance comparison table
comparison = pd.DataFrame({
    'Step': [r['step'] for r in knn_results],
    'Num_Features': [r['num_features'] for r in knn_results],
    'KNN_Accuracy': [r['accuracy'] for r in knn_results],
    'NB_Accuracy': [r['accuracy'] for r in nb_results],
    'Feature_Group': [r['feature_label'] for r in knn_results]
})

# Plot model performance
plt.figure(figsize=(10, 5))
plt.plot(comparison['Step'], comparison['KNN_Accuracy'], marker='o', label='KNN')
plt.plot(comparison['Step'], comparison['NB_Accuracy'], marker='o', label='Naive Bayes')
plt.xticks(comparison['Step'], labels=[f"Step {s}" for s in comparison['Step']])
plt.xlabel('Feature Addition Stage')
plt.ylabel('Accuracy')
plt.title('Model Performance at Each Feature Stage (HTRU_2 Dataset)')
plt.legend()
plt.grid(True)
plt.savefig('performance_comparison_htru2.png')
plt.close()

# Final results
print("\nPerformance Comparison:")
print(comparison.to_string(index=False))
print("\nFiles saved:")
print("- Confusion matrices for each step/model")
print("- Performance comparison plot: performance_comparison_htru2.png")
