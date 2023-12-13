import igraph
class Walktrap:
    def __init__(self, G):
        self._G = G
        self.ig_graph = igraph.Graph(directed=False)  
        # 添加节点  
        self.ig_graph.add_vertices(len(self._G.nodes()))  
  
        # 添加边  
        edges = [tuple(edge) for edge in self._G.edges()]  
        self.ig_graph.add_edges(edges) 

    def excuate(self):
        clusters = self.ig_graph.community_walktrap()
        output = []
        for  community in clusters.as_clustering():              
            output.append(community) 
        return output
    
if __name__ == '__main__':
    from utils import process_original_data, nonoverlap_modylarity_score
    G = process_original_data()
    walktrap = Walktrap(G)
    clusters = walktrap.excuate()
    print(clusters)
    print(nonoverlap_modylarity_score(G, clusters))