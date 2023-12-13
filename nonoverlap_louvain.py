from community import community_louvain


class Louvain:
    def __init__(self, G):
        self._G = G

    def excuate(self):
        partition = community_louvain.best_partition(self._G)
        return partition

if __name__ == '__main__':
    from utils import process_original_data_str_key
    import networkx as nx
    import community
    G = process_original_data_str_key()
    print(nx.info(G))
    louvain = Louvain(G)
    community_dict = louvain.excuate()
    print(community_dict)
    print('Modularity: ', community.modularity(community_dict, G))