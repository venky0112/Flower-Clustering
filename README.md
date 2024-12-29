# Flower Clustering Using K-Means

This repository demonstrates how to perform clustering on the Iris dataset using the K-Means algorithm. The project includes data preprocessing, clustering implementation, and visualization of the clustering results.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Snippets](#code-snippets)
- [Results](#results)
- [Contributing](#contributing)

## Overview

Clustering is an unsupervised learning technique used to group similar data points. This project focuses on clustering the famous Iris dataset using the K-Means algorithm, a popular clustering method.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/venky0112/Flower-Clustering.git

# Navigate to the project directory
cd Flower-Clustering

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the notebook to experiment with K-Means clustering and visualize the results:

```bash
# Open the Jupyter Notebook
jupyter notebook k_means_clustering.ipynb
```

Follow the instructions in the notebook to execute each cell and observe the output.

## Code Snippets

Below are detailed explanations of the code snippets used in the implementation:

### **Data Preprocessing**

```python
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.iloc[:, :-1])
```

**Explanation:**

1. **Loading the Iris Dataset:**
   - `load_iris` loads the Iris dataset, which includes measurements of petal and sepal lengths and widths for three flower species.
   - A DataFrame is created with the feature values and species labels.

2. **Standardizing Features:**
   - Standardizing ensures that features are on the same scale, which is important for distance-based algorithms like K-Means.
   - `StandardScaler` transforms the features to have zero mean and unit variance.

---

### **K-Means Clustering Implementation**

#### Step 1: Determine the Optimal Number of Clusters

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
```

**Explanation:**

1. **Inertia:**
   - `Inertia` measures how tightly the data points are clustered around centroids. Lower values indicate better clustering.

2. **Elbow Method:**
   - The elbow method helps identify the optimal number of clusters by looking for the "elbow point" in the plot, where the rate of inertia decrease slows.

---

#### Step 2: Apply K-Means with Optimal Clusters

```python
# Apply K-Means with optimal clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_
```

**Explanation:**

1. **Choosing Number of Clusters:**
   - Based on the elbow method, the number of clusters (`n_clusters=3`) is chosen.

2. **Fitting the Model:**
   - The K-Means algorithm assigns each data point to one of the three clusters based on similarity.

3. **Cluster Labels:**
   - `labels` contains the cluster assignment for each data point (e.g., cluster 0, 1, or 2).

---

### **Visualization of Clusters**

```python
from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Scatter plot of clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.title('Iris Flower Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

**Explanation:**

1. **Dimensionality Reduction:**
   - PCA reduces the dataset to two dimensions for visualization, preserving the maximum variance.

2. **Visualization:**
   - A scatter plot displays the clusters in a 2D space, with colors indicating cluster assignments.

---

## Results

By running the notebook, you will:

1. Load and preprocess the Iris dataset.
2. Determine the optimal number of clusters using the elbow method.
3. Apply K-Means clustering and visualize the clusters in reduced dimensions.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.


