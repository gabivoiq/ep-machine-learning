import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from src.utils.printers import print_clustering_results

# NOTE: You can play around with these values
CLUSTERS = 4
SAMPLES = 300
CLUSTERS_STD = 1


# Evaluates the performance of the clustering algorithm
def evaluate_clusters(X, y_pred):
    return silhouette_score(X, y_pred)


# Plots the clusters and their centres
def plot_clusters(model, X, y_pred):
    # Fetch the centers of the clusters
    center = model.cluster_centers_

    # Pretty print the resulting clusters together with their centers
    list_col = []
    for val in y_pred:
        if val == 0:
            list_col.append('purple')
        elif val == 1:
            list_col.append('yellow')
        elif val == 2:
            list_col.append('blue')
        elif val == 3:
            list_col.append('green')

    plt.title("CLUSTER_STD = %d" % CLUSTERS_STD)
    plt.scatter(X[:, 0], X[:, 1], c=list_col, s=50)
    plt.scatter(center[:, 0], center[:, 1], color='black')
    plt.show()


# Performs clustering on a sample dataset
def perform_clustering():
    # Generate a dataset
    X, _ = make_blobs(n_samples=SAMPLES, centers=CLUSTERS, cluster_std=CLUSTERS_STD, random_state=13)

    # Plot the freshly generated blobs
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.show()

    # Create a clustering model (less important: a K-means clustering algorithm will be used)
    model = KMeans(n_clusters=CLUSTERS, random_state=13)

    # Fit the data
    model.fit(X)

    # Predict a cluster number for each point
    y_pred = model.predict(X)

    # Compute the silhouette score of the clusters and print it
    score = evaluate_clusters(X, y_pred)
    print_clustering_results(score)

    # Plot the clusters and their centres
    plot_clusters(model, X, y_pred)
