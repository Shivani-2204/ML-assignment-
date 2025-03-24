import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\KIIT\Desktop\AD lab\kmeans - kmeans_blobs.csv"  # Path to your dataset
data = pd.read_csv(file_path)

print("Dataset preview:")
print(data.head())

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

X = data.select_dtypes(include=[np.number])
X_normalized = normalize(X).values

def initialize_centroids(X, k):
    """ Randomly initialize k centroids """
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """ Assign points to the nearest centroid """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """ Update centroids as the mean of the cluster points """
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        # Convergence check
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    
    return centroids, labels

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, k in enumerate([2, 3]):
    centroids, labels = kmeans(X_normalized, k)

    # Plot the clusters
    axes[idx].scatter(X_normalized[:, 0], X_normalized[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    axes[idx].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Centroids', s=200)
    axes[idx].set_title(f'K-means Clustering (k={k})')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].legend()

plt.tight_layout()
plt.show()
