from sklearn.cluster import SpectralClustering
import networkx as nx
import numpy as np


class SpectralCluster:
    def __init__(self, G):
        self._G = G

    def excuate(self):
        num_communities = 10
        sc = SpectralClustering(n_clusters=num_communities, affinity='nearest_neighbors', random_state=42)

        adj_matrix = nx.to_numpy_array(self._G)
        labels = sc.fit_predict(adj_matrix)
        communities = [list(np.where(labels == i)[0]) for i in range(num_communities)]

        return communities
