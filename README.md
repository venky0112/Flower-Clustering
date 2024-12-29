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

Below are some key snippets from the implementation:

### Data Preprocessing

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

### K-Means Clustering Implementation

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

# Apply K-Means with optimal clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_
```

### Visualization of Clusters

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

## Results

By running the notebook, you will:

1. Preprocess and scale the flower dataset.
2. Determine the optimal number of clusters using the elbow method.
3. Apply K-Means clustering and visualize the clusters in reduced dimensions.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.
