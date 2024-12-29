# Flower Clustering Using K-Means

This repository demonstrates how to perform clustering on flower datasets using the K-Means algorithm. The project includes data preprocessing, clustering implementation, and visualization of the clustering results.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Snippets](#code-snippets)
- [Results](#results)
- [Contributing](#contributing)

## Overview

Clustering is an unsupervised learning technique used to group similar data points. This project focuses on clustering flower data using the K-Means algorithm, a popular clustering method.

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
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('flowers.csv')

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

**Explanation:**

1. **Importing Libraries:**
   - `pandas` is used for data manipulation and loading the dataset.
   - `numpy` provides efficient numerical computations.
   - `StandardScaler` from `sklearn.preprocessing` ensures that features have zero mean and unit variance, which is crucial for K-Means clustering.

2. **Loading the Dataset:**
   - `flowers.csv` is loaded into a DataFrame. It is assumed that the dataset contains numerical features for clustering.

3. **Feature Scaling:**
   - The `StandardScaler` is used to scale the features to a standard normal distribution, preventing any single feature from dominating the clustering process.

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

1. **What is Inertia?**
   - `Inertia` measures how well the clusters are formed. It is the sum of squared distances between data points and their respective cluster centroids.

2. **Elbow Method:**
   - The elbow method is used to find the optimal number of clusters (`k`).
   - The range of `k` values is tested (from 1 to 10), and the corresponding `inertia` values are plotted.
   - The "elbow point" on the plot indicates the optimal number of clusters, where the inertia sharply decreases.

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
   - Based on the elbow method, `n_clusters=3` is chosen for this example.

2. **Clustering Process:**
   - The `fit` method applies the K-Means algorithm to the scaled dataset.
   - `labels` contain the cluster assignments for each data point (e.g., `0`, `1`, `2`). These labels can be used to analyze and visualize the clusters.

---

### **Visualization of Clusters**

```python
from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Scatter plot of clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.title('Flower Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

**Explanation:**

1. **Dimensionality Reduction:**
   - PCA (Principal Component Analysis) reduces the dataset's dimensions to 2 for visualization while preserving as much variance as possible.

2. **Visualization:**
   - A scatter plot is generated to visualize the clusters in two-dimensional space.
   - The `c=labels` argument colors the points according to their cluster assignments.

---

## Results

By running the notebook, you will:

1. Preprocess and scale the flower dataset.
2. Determine the optimal number of clusters using the elbow method.
3. Apply K-Means clustering and visualize the clusters in reduced dimensions.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

