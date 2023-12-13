from sklearn.cluster import KMeans 
import networkx as nx  
from gensim.models import Word2Vec  
import numpy as np
from tqdm import tqdm
import random
from sklearn.neighbors import NearestNeighbors
class DeepWalk:
    def __init__(self, G, SubG):
        self._G = G
        self._SubG = SubG
    
    def deepwalk(self, num_walks, walk_length, embed_size):  
        walks = []  
        # 对每个节点执行多次随机游走
        for i in tqdm(range(num_walks)):  
            for node in self._SubG.nodes():  
                walk = self.random_walk(walk_length, start=node)  
                walks.append(walk)  
        # Gensim 库中的 Word2Vec 模型来学习节点的向量表示,返回训练好的 Word2Vec 模型，该模型包含了学习到的节点向量
        model = Word2Vec(walks, vector_size=embed_size, window=5, min_count=0, sg=1, workers=2)  
        return model  

    # 定义随机游走函数  根据设定的步数执行随机游走，并返回游走的路径
    def random_walk(self, walk_length, start):  
        walk = [start]  
        for _ in range(walk_length - 1):  
            current_node = walk[-1]  
            neighbors = list(self._G.neighbors(current_node))  
            if len(neighbors) > 0:  
                walk.append(random.choice(neighbors))  
            else:  
                break  
        return [str(node) for node in walk]  


    def excuate(self, num_walks, walk_length, embed_size, num_clusters):
        dw_model = self.deepwalk(num_walks, walk_length, embed_size) 
        node_vectors = {}
        for i in dw_model.wv.key_to_index.keys():
            node_vectors[i] = dw_model.wv[i]
        sorted_dict = dict(sorted(node_vectors.items(), key=lambda x: int(x[0])))
        embed = []
        for i in sorted_dict.keys():
            embed.append(sorted_dict[i])
        embed = np.array(embed)

        kmeans = KMeans(n_clusters = num_clusters, init= "random", n_init = 3, random_state = 1010)  
        kmeans.fit(embed)  
        # 获取k-means聚类结果  
        cluster_labels = kmeans.labels_ 
        output_data = {}
        for i in range(len(sorted_dict.keys())):
            output_data[int(list(sorted_dict.keys())[i])] = [cluster_labels.tolist()[i]]
        max_cluster = max(cluster_labels.tolist())
        for i in range(len(self._G.nodes())):
            if i not in output_data.keys():
                output_data[i] = [max_cluster]
                max_cluster = max_cluster + 1
        output_data =  dict(sorted(output_data.items(), key=lambda x: int(x[0])))
 
        return output_data