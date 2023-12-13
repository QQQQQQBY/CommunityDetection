from networkx.algorithms.community import label_propagation_communities


class LPA:
    def __init__(self, G):
        self._G = G

    def excuate(self):
        communities = list(label_propagation_communities(self._G))
        return communities

if __name__ == '__main__':
    from utils import process_original_data_str_key, nonoverlap_modylarity_score
    import networkx as nx
    G = process_original_data_str_key()
    print(nx.info(G))
    lpa = LPA(G)
    community_dict = lpa.excuate()
    print('Modularity: ', nonoverlap_modylarity_score(G, community_dict))