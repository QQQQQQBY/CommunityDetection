import argparse
from overlap_slpa import SLPA
from nonoverlap_louvain import Louvain
from overlap_scan import SCAN
from nonoverlap_deepwalk import DeepWalk
from utils import overlap_modylarity_score, nonoverlap_modylarity_score, divide_isolated_node, process_original_data, \
    louvain_modylarity_score
from nonoverlap_girvan_newman import GirvanNewman
from nonoverlap_lfm import LFM
from nonoverlap_clusternet import NonoverlapClusterNet
from overlap_lpa import LPA

from nonoverlap_infomap import InfoMap
from nonoverlap_walktrap import Walktrap
from nonoverlap_multilevel import Multilevel
from nonoverlap_fast import Fast
from nonoverlap_spectral_cluster import SpectralCluster
import torch
import numpy as np
import random


# 划分孤立节点的方法
class CommunityDetection_Isolated:
    def __init__(self, G, SubG, args):
        self.slpa = SLPA(G, args.T, args.r, SubG)
        self.deepwalk = DeepWalk(G, SubG)

        self.nonoverlap_cluster_net = NonoverlapClusterNet(SubG)

    def fusion_isolated_nodes(self, G, non_isolate_community):
        _, isolated_nodes, subgraph_to_original_mapping = divide_isolated_node(G)
        community_dict = {subgraph_to_original_mapping[i]: [non_isolate_community[i]] for i in
                          range(len(non_isolate_community))}
        all_values = [value for values in community_dict.values() for value in values]
        max_value = max(all_values)
        isolated_communities = {isolated_node: [i + max_value + 1] for i, isolated_node in
                                enumerate(isolated_nodes)}
        community_dict.update(isolated_communities)
        return community_dict

    def excuate(self, args):
        communities_slpa = self.slpa.excuate()
        communities_deepwalk = self.deepwalk.excuate(args.num_walks, args.walk_length, args.embed_size,
                                                     args.num_clusters)

        communities_nonoverlap_cluster_net = self.nonoverlap_cluster_net.excuate()
        return communities_slpa, communities_deepwalk, communities_nonoverlap_cluster_net


# 不划分孤立节点的方法
class CommunityDetection_NonIsolated:
    def __init__(self, G, SubG, args):
        self.walktrap = Walktrap(G)
        self.fast = Fast(G)
        self.louvain = Louvain(G)
        self.lpa = LPA(G)
        self.multilevel = Multilevel(G)
        self.infomap = InfoMap(G)
        # 运行时间较长
        # self.lfm = LFM(G)
        # self.girvan_newman = GirvanNewman(G)
        # self.spectral_cluster = SpectralCluster(G)
        # self.scan = SCAN(G, args.epsilon, args.mu)

    def excuate(self, args):
        communities_fast = self.fast.excuate()
        communities_walktrap = self.walktrap.excuate()
        communities_multilevel = self.multilevel.excuate()
        communities_louvain = self.louvain.excuate()
        communities_lpa = self.lpa.excuate()
        communities_infomap = self.infomap.excuate()

        # 运行时间较长
        # communities_lfm = self.lfm.execute()
        # communities_girvan_newman = self.girvan_newman.excuate()
        # communities_spectral_cluster = self.spectral_cluster.excuate()
        # communities_scan = self.scan.excuate()

        return communities_fast, communities_walktrap, communities_multilevel, communities_louvain, communities_lpa, communities_infomap


def ensemble_voting(community_dicts):
    max_modularity = float('-inf')

    for community_dict in community_dicts.keys():
        modularity_score = community_dicts[community_dict]

        if modularity_score > max_modularity:
            max_modularity = modularity_score
            key = community_dict
    return max_modularity, key


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 设置所有方法所需参数

if __name__ == "__main__":
    # 定义相关参数
    parser = argparse.ArgumentParser(
        description='Community Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # GNN方法所需参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    # SLPA方法所需参数
    parser.add_argument('--T', default=20, type=int, help='Number of iterations')
    parser.add_argument('--r', default=0.2, type=float, help='Community threshold')

    # deepwalk方法所需参数
    parser.add_argument('--num_walks', default=10, type=int, help='Number of random walks')
    parser.add_argument('--walk_length', default=80, type=int, help='The length of each random walk')
    parser.add_argument('--embed_size', default=64, type=int, help='Dimension of the node vector')
    parser.add_argument('--num_clusters', default=20, type=int, help='Kmeans clustering')

    # scan方法所需参数
    parser.add_argument('--epsilon', default=0.5, type=float, help='similarity')
    parser.add_argument('--mu', default=3, type=float, help='Number of neighbors')

    parser.add_argument('--seed', default=3, type=float, help='Seed')
    args = parser.parse_args()
    setup_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    G = process_original_data()
    SubG, isolated_nodes, subgraph_to_original_mapping = divide_isolated_node(G)
    # 实例化
    Isolated_model = CommunityDetection_Isolated(G, SubG, args)
    NonIsolated_model = CommunityDetection_NonIsolated(G, SubG, args)

    # 社区划分
    communities_slpa, communities_deepwalk, communities_nonoverlap_cluster_net = Isolated_model.excuate(args)
    communities_fast, communities_walktrap, communities_multilevel, communities_louvain, communities_lpa, communities_infomap = NonIsolated_model.excuate(
        args)

    # 模块度计算
    modularity_scores = {}
    modularity_scores['modularity_slpa'] = overlap_modylarity_score(communities_slpa, G)
    modularity_scores['modularity_deepwalk'] = overlap_modylarity_score(communities_deepwalk, G)
    modularity_scores['modularity_nonoverlap_cluster_net'] = communities_nonoverlap_cluster_net

    modularity_scores['modularity_fast'] = nonoverlap_modylarity_score(G, communities_fast)
    modularity_scores['modularity_walktrap'] = nonoverlap_modylarity_score(G, communities_walktrap)
    modularity_scores['modularity_multilevel'] = nonoverlap_modylarity_score(G, communities_multilevel)
    modularity_scores['modularity_louvain'] = louvain_modylarity_score(G, communities_louvain)
    modularity_scores['modularity_lpa'] = nonoverlap_modylarity_score(G, communities_lpa)
    modularity_scores['modularity_infomap'] = overlap_modylarity_score(communities_infomap, G)

    # 运行时间较长
    # modularity_scores['modularity_lfm'] = nonoverlap_modylarity_score(communities_lfm)
    # modularity_scores['modularity_girvan_newman'] = nonoverlap_modylarity_score(communities_girvan_newman)
    # modularity_scores['modularity_spectral_cluster'] = nonoverlap_modylarity_score(communities_spectral_cluster)
    max_modularity, key = ensemble_voting(modularity_scores)
    print(f'max modylarity: {key}:{max_modularity}')
