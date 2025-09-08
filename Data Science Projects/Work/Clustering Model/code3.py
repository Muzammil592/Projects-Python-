import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------- Load Heart Failure Dataset --------------------
heart_failure = pd.read_csv("D:/IDS ASSIGNMENT/Datasets/heart+failure+clinical+records/heart_failure_clinical_records_dataset.csv")

# -------------------- Bivariate Analysis --------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=heart_failure, x="age", y="ejection_fraction", hue="DEATH_EVENT", palette="viridis")
plt.title("Heart Failure: Age vs Ejection Fraction by Death Event")
plt.xlabel("Age")
plt.ylabel("Ejection Fraction (%)")
plt.savefig("bivariate_heart_failure.png")
plt.close()

# -------------------- Clustering (K-Means) --------------------
# Select relevant features for clustering
features = heart_failure[['age', 'ejection_fraction']]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow Method to determine optimal K
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_features)
    wcss.append(km.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method - Optimal K for Heart Failure Dataset")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.xticks(K_range)
plt.savefig("elbow_heart_failure.png")
plt.close()

# Apply KMeans with optimal K (likely 2 or 3 based on elbow method)
optimal_k = 2  # You might want to adjust this after seeing the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
heart_failure['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize Clusters (using the same features as bivariate analysis)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=heart_failure, x="age", y="ejection_fraction", hue="Cluster", palette="Set1")
plt.title(f"K-Means Clusters (K={optimal_k}) on Heart Failure Dataset")
plt.xlabel("Age")
plt.ylabel("Ejection Fraction (%)")
plt.savefig("kmeans_heart_failure.png")
plt.close()

# -------------------- Summary --------------------
print("✅ Heart Failure Dataset Analysis Complete:")
print("- Bivariate Plots: bivariate_heart_failure.png and bivariate_heart_failure2.png")
print("- Elbow Curve: elbow_heart_failure.png")
print("- Cluster Plots: kmeans_heart_failure.png and kmeans_heart_failure2.png")

# Additional analysis: Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = heart_failure.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Heart Failure Features")
plt.tight_layout()
plt.savefig("correlation_matrix_heart_failure.png")
plt.close()