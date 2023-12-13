import networkx as nx
import pandas as pd
from networkx.algorithms.community.quality import modularity
import collections
import numpy as np
from tqdm import tqdm
import re


# 处理原始数据
def process_original_data():
    repost_data = pd.read_csv(
        './Data/repost.data.csv', delimiter="\t")
    post_data = pd.read_csv(
        './Data/post.data.csv', delimiter="\t")
    user_data = pd.read_csv(
        './Data/user.data.csv', delimiter="\t")

    # 用户id的映射
    user_index_to_uid = list(user_data['userid'])
    uid_to_user_index = {x: i for i, x in enumerate(user_index_to_uid)}

    G = nx.Graph()
    # 为图结构添加节点
    for index, user in user_data.iterrows():
        G.add_node(index)
    # 处理原始数据边信息
    for _, row in repost_data.iterrows():
        source = row['userid']
        mentions = re.findall(r'//@(\d+):', str(row['content']))
        dst = row['userid']
        last = row['rootuserid']
        for mention_index in range(len(mentions)):
            mention = mentions[mention_index]
            user_pair = (uid_to_user_index[int(mention)], uid_to_user_index[dst])
            dst = int(mention)
            # 为图结构添加边
            G.add_edge(*user_pair)
        G.add_edge(uid_to_user_index[last], uid_to_user_index[dst])

    return G 


# 划分孤立节点
def divide_isolated_node(G):
    non_isolated_nodes = [node for node in G.nodes if G.degree(node) > 0]
    isolated_nodes = [node for node in G.nodes if G.degree(node) == 0]
    SubG = nx.Graph()
    SubG.add_nodes_from(non_isolated_nodes)
    SubG.add_edges_from(G.edges)
    # 创建子图，只包含非孤立节点
    # subgraph = G.subgraph(non_isolated_nodes)
    subgraph_index_to_ori = list(SubG.nodes())
    subgraph_to_original_mapping = {i: node for i, node in enumerate(subgraph_index_to_ori)}
    return SubG, isolated_nodes, subgraph_to_original_mapping


# 非重叠社区评价指标
# communities 格式list类型， 长度为社区数量，元素为每个社区的节点index
def nonoverlap_modylarity_score(G, communities):
    modularity_score = modularity(G, communities)
    return modularity_score

def louvain_modylarity_score(G, communities):
    import community 
    return community.modularity(communities, G)


# 重叠社区评价指标


def convert_to_community_lists(node_community_dict):
    community_lists = {}

    for node, communities in node_community_dict.items():
        for community in communities:
            if community in community_lists:
                community_lists[community].append(int(node))
            else:
                community_lists[community] = [int(node)]
    # 排序
    community_dict = dict(sorted(community_lists.items(), key=lambda x: int(x[0])))
    result = list(community_dict.values())

    return result


# partition 是节点属于的社区index， 输入格式为字典
def overlap_modylarity_score(partition, G):
    cover = convert_to_community_lists(partition)
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # 存储每个节点所在的社区
    vertex_community = collections.defaultdict(lambda: set())
    # i为社区编号(第几个社区) c为该社区中拥有的节点
    for i, c in enumerate(cover):
        # v为社区中的某一个节点
        for v in c:
            # 根据节点v统计他所在的社区i有哪些
            vertex_community[v].add(i)
    total = 0.0
    for c in tqdm(cover):
        for i in c:
            # o_i表示i节点所同时属于的社区数目
            o_i = len(vertex_community[i])
            # k_i表示i节点的度数(所关联的边数)
            k_i = len(G[i])
            for j in c:
                t = 0.0
                # o_j表示j节点所同时属于的社区数目
                o_j = len(vertex_community[j])
                # k_j表示j节点的度数(所关联的边数)
                k_j = len(G[j])
                if G.has_edge(i, j):
                    t += 1.0 / (o_i * o_j)
                t -= k_i * k_j / (2 * m * o_i * o_j)
                total += t
    return total / (2 * m)


def process_original_data_str_key():
    repost_data = pd.read_csv(
        './Data/repost.data.csv', delimiter="\t")
    post_data = pd.read_csv(
        './Data/post.data.csv', delimiter="\t")
    user_data = pd.read_csv(
        './Data/user.data.csv', delimiter="\t")

    G = nx.Graph()
    for _, user in user_data.iterrows():
        G.add_node(user['userid'])

    for _, row in repost_data.iterrows():
        source = row['userid']
        mentions = re.findall(r'//@(\d+):', str(row['content']))
        dst = row['userid']
        last = row['rootuserid']
        for mention_index in range(len(mentions)):
            mention = mentions[mention_index]
            # user_pair = (src, int(mention))
            user_pair = (int(mention), dst)
            dst = int(mention)
            G.add_edge(*user_pair)
        G.add_edge(last, dst)
    return G 