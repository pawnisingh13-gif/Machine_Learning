# K-Means Clustering Example

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
X = np.array([
    [1,2],[2,3],[3,4],
    [8,7],[9,8],[10,9]
])

# Model
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Labels
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Centroids:", centroids)

# Visualization
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='x')
plt.title("K-Means Clustering")
plt.show()
