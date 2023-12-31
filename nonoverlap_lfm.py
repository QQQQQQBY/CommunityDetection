import random
import networkx as nx
import matplotlib.pyplot as plt


class Community():

    def __init__(self, G, alpha=1.0):
        self._G = G
        self._alpha = alpha
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0

    def add_node(self, node):
        neighbors = set(self._G.neighbors(node))
        # print("添加令居节点",neighbors , self._nodes,neighbors & self._nodes)
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        # print("node_k_out",node_k_out)
        self._nodes.add(node)
        self._k_in += 2 * node_k_in
        self._k_out = self._k_out + node_k_out - node_k_in

    def remove_node(self, node):
        neighbors = set(self._G.neighbors(node))
        community_nodes = self._nodes
        # print("community_nodes",community_nodes)
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2 * node_k_in
        self._k_out = self._k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):  # fitness适应度
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)  # vertex顶点
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha  # 幂次
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        neighbors = set(self._G.neighbors(node))
        new_k_in = self._k_in
        new_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        return new_fitness - old_fitness

    def recalculate(self):
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._G.neighbors(node)) - self._nodes)
        return neighbors

    def get_fitness(self):
        return float(self._k_in) / ((self._k_in + self._k_out) ** self._alpha)


class LFM:
    def __init__(self, G):
        self._G = G
        alpha = 1.0
        self._alpha = alpha

    def execute(self):
        communities = []
        node_not_include = list(self._G.node.keys())
        while (len(node_not_include) != 0):
            c = Community(self._G, self._alpha)
            seed = random.choice(node_not_include)
            c.add_node(seed)
            to_be_examined = c.get_neighbors()
            while (to_be_examined):
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)
                    m[node] = fitness
                to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]

                # stop condition
                if (to_be_add[1] < 0.0):
                    break
                c.add_node(to_be_add[0])
                to_be_remove = c.recalculate()
                while (to_be_remove != None):
                    c.remove_node(to_be_remove)
                    to_be_remove = c.recalculate()

                to_be_examined = c.get_neighbors()

            for node in c._nodes:
                if (node in node_not_include):
                    node_not_include.remove(node)
            communities.append(c._nodes)
        return communities
