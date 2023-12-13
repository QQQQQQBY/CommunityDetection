import networkx as nx
class Fast:
    def __init__(self, G):
         self._G = G
    
    def excuate(self):
        communities = list(nx.algorithms.community.modularity_max.greedy_modularity_communities(self._G))   
        return communities

