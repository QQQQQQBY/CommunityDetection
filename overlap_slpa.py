import numpy as np
import networkx as nx
import torch
import random
import collections

class SLPA:
    def __init__(self, G, T, r, SubG):
        """
        :param G:图
        :param T: 迭代次数T
        :param r:满足社区次数要求的阈值r
        """
        self._SubG = SubG
        self._G = G
        self._n = len(SubG.nodes(False))  # 节点数目
        self._T = T
        self._r = r

    def excuate_slpa(self):
        # 处理图结构
        G, i_to_original_node = self.process_subg()
        # 节点存储器初始化
        node_memory = []
        for i in range(self._n):
            node_memory.append({i: 1})

        # 算法迭代过程
        for t in range(self._T):
            # 任意选择一个监听器
            # np.random.permutation()：随机排列序列
            order = [x for x in np.random.permutation(self._n)]
            for i in order:
                label_list = {}
                # 从speaker中选择一个标签传播到listener
                for j in G.neighbors(i):
                    sum_label = sum(node_memory[j].values())
                    label = list(node_memory[j].keys())[np.random.multinomial(
                        1, [float(c) / sum_label for c in node_memory[j].values()]).argmax()]
                    label_list[label] = label_list.setdefault(label, 0) + 1
                # listener选择一个最流行的标签添加到内存中
                max_v = max(label_list.values())
                # selected_label = max(label_list, key=label_list.get)
                selected_label = random.choice([item[0] for item in label_list.items() if item[1] == max_v])
                # setdefault如果键不存在于字典中，将会添加键并将值设为默认值。
                node_memory[i][selected_label] = node_memory[i].setdefault(selected_label, 0) + 1

        # 根据阈值threshold删除不符合条件的标签
        for memory in node_memory:
            sum_label = sum(memory.values())
            threshold_num = sum_label * self._r
            for k, v in list(memory.items()):
                if v < threshold_num:
                    del memory[k]

        communities = collections.defaultdict(lambda: list())
        # 扫描memory中的记录标签，相同标签的节点加入同一个社区中
        for primary, change in enumerate(node_memory):
            for label in change.keys():
                communities[label].append(primary)
        # 返回值是个数据字典，value以集合的形式存在
        return communities.values(), i_to_original_node

    def process_subg(self):
        G = nx.Graph()
        node = list(set(np.array(self._SubG.edges).reshape(-1).tolist()))
        subgraph_to_original_mapping = {node: i for i, node in enumerate(node)}
        i_to_original_node = {i: node for i, node in enumerate(node)}
        s_new = []
        t_new = []
        for edge in list(self._SubG.edges):
            s = edge[0]
            t = edge[1]
            if s in subgraph_to_original_mapping.keys():
                s_new.append(subgraph_to_original_mapping[s])
            if t in subgraph_to_original_mapping.keys():
                t_new.append(subgraph_to_original_mapping[t])
        edge_index_new = []
        edge_index_new.append(s_new)
        edge_index_new.append(t_new)
        edge_index_new = torch.tensor(edge_index_new)
        G.add_edges_from(edge_index_new.T.tolist())
        G.add_nodes_from(range(len(subgraph_to_original_mapping)))
        return G, i_to_original_node

    def excuate(self):
        communities, i_to_original_node = self.excuate_slpa()
        output_data = {}
        for num, coms in enumerate(communities):
            for com in coms:
                if i_to_original_node[com] not in output_data.keys():
                    output_data[i_to_original_node[com]] = []
                output_data[i_to_original_node[com]].append(num)
        max_cluster = len(communities)
        for i in range(len(self._G.nodes())):
            if i not in output_data.keys():
                output_data[i] = [max_cluster]
                max_cluster = max_cluster + 1
        output_data =  dict(sorted(output_data.items(), key=lambda x: int(x[0])))
        return output_data