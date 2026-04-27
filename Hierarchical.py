# Hierarchical Clustering Example

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Dataset
X = np.array([
    [1,2],[2,3],[3,4],
    [8,7],[9,8],[10,9]
])

# Linkage
linked = linkage(X, method='ward')

# Dendrogram
plt.figure(figsize=(6,4))
dendrogram(linked)
plt.title("Hierarchical Clustering")
plt.show()
