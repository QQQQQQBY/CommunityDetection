import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import sklearn
import sklearn.cluster
import scipy.sparse as sp
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import networkx as nx
import pandas as pd
import re
import community
import random
from utils import overlap_modylarity_score, nonoverlap_modylarity_score, divide_isolated_node, process_original_data


class arguments():
    def __init__(self):
        self.no_cuda = True
        self.seed = random.randint(1, 1000)
        self.lr = 0.001
        self.weight_decay = 5e-4
        self.hidden = 512
        self.embed_dim = 50
        self.dropout = 0.5
        self.K = 10
        self.clustertemp = 100
        self.train_iters = 501
        self.num_cluster_iter = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def mx_to_sparse_tensor(mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 加载图数据集
def load_data(G):
    """Load network (graph)"""
    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = mx_to_sparse_tensor(adj)
    features = torch.eye(len(G.nodes())).to_sparse()
    return adj, features


# 获得模块度矩阵
def make_modularity_matrix(adj):
    adj = adj * (torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))  # 非对角线位置为1
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees @ degrees.t() / adj.sum()
    return mod


def loss_modularity(r, bin_adj, mod):
    bin_adj_nodiag = bin_adj * (
            torch.ones(bin_adj.shape[0], bin_adj.shape[0]).cuda() - torch.eye(bin_adj.shape[0]).cuda())
    return (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()


def cluster(data, k, num_iter, init=None, cluster_temp=5):
    data = torch.diag(1. / torch.norm(data, p=2, dim=1)) @ data
    if init is None:
        data_np = data.detach().numpy()
        norm = (data_np ** 2).sum(axis=1)
        init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        init = torch.tensor(init, requires_grad=True).cuda()
        if num_iter == 0:
            return init
    mu = init
    for t in range(num_iter):
        dist = data @ mu.t()
        r = torch.softmax(cluster_temp * dist, 1)
        cluster_r = r.sum(dim=0)
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp * dist, 1)
    return mu, r, dist


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # XWA
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GCNClusterNet(nn.Module):

    def __init__(self, nfeat, nhid, nout, dropout, K, cluster_temp):
        super(GCNClusterNet, self).__init__()
        self.GCN = GCN(nfeat, nhid, nout, dropout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init = torch.rand(self.K, nout).cuda()

    def forward(self, x, adj, num_iter=1, mu=None):
        embeds = self.GCN(x, adj)
        mu_init, _, _ = cluster(embeds, self.K, num_iter, init=mu, cluster_temp=self.cluster_temp)
        mu, r, dist = cluster(embeds, self.K, num_iter, init=mu_init.detach().clone(), cluster_temp=self.cluster_temp)
        return mu, r, embeds, dist


EPS = 1e-15

args = arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()
num_cluster_iter = args.num_cluster_iter
K = args.K
seed = args.seed
setup_seed(seed)

losses = []
losses_test = []
best_train_val = 100
curr_test_loss = 100
best_community_modularity = -1



class NonoverlapClusterNet():
    def __init__(self, G) -> None:
        self.G = process_original_data()

    def excuate(self):
        SubG,isolated_nodes, subgraph_to_original_mapping = divide_isolated_node(self.G)
        adj_all, features = load_data(SubG)
        bin_adj_all = (adj_all.to_dense() > 0).float()
        test_object = make_modularity_matrix(bin_adj_all)
        nfeat = features.shape[1]
        adj_all = adj_all.cuda()
        features = features.cuda()
        bin_adj_all = bin_adj_all.cuda()
        test_object = test_object.cuda()
        best_community = None

        self.model_cluster = GCNClusterNet(nfeat=nfeat,
                                           nhid=args.hidden,
                                           nout=args.embed_dim,
                                           dropout=args.dropout,
                                           K=args.K,
                                           cluster_temp=args.clustertemp)

        self.optimizer = optim.Adam(self.model_cluster.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        self.model_cluster.cuda()
        best_community_modularity = -1
        mu = torch.rand(K, args.embed_dim).cuda()
        for t in range(args.train_iters):
            mu, r, embeds, dist = self.model_cluster(features, adj_all, num_cluster_iter, mu=mu)
            loss = loss_modularity(r, bin_adj_all, test_object)
            print(loss)
            loss = -loss
            self.optimizer.zero_grad()
            loss.backward()

            r = torch.softmax(100 * r, dim=1)
            loss_test = loss_modularity(r, bin_adj_all, test_object)
            node_communities = torch.argmax(r, dim=1).tolist()
            community_dict = {subgraph_to_original_mapping[i]: [node_communities[i]][0] for i in
                              range(len(node_communities))}
            # all_values = [value for values in community_dict.values() for value in values]
            max_value = K + 10
            isolated_communities = {isolated_node: [i + max_value + 1][0] for i, isolated_node in
                                    enumerate(isolated_nodes)}
            community_dict.update(isolated_communities)
            modularity_score = community.modularity(community_dict, self.G)

            if modularity_score > best_community_modularity:
                best_community_modularity = modularity_score
                best_community = community_dict
                best_train_val = loss.item()
                curr_test_loss = loss_test.item()
                output_data = {}
                torch.save(self.model_cluster.state_dict(), './save.pt')
                if best_community_modularity > 0.595:
                    for user_id, community_id in community_dict.items():
                        if user_id not in output_data:
                            output_data[user_id] = []
                        output_data[user_id].append([community_id])
                    import json
                    with open('./submission_{}.json'.format(best_community_modularity), 'w') as json_file:
                        json.dump(output_data, json_file)
            print('Community modularity: ', modularity_score, 'Best Community modularity: ', best_community_modularity)
            log = 'Iterations: {:03d}, ClusterNet modularity: {:.4f}'
            print(log.format(t, curr_test_loss))
            losses.append(loss.item())
            self.optimizer.step()

        return best_community_modularity


if __name__ == "__main__":
    nonoverlap_cluster_net = NonoverlapClusterNet(None)
    best_community = nonoverlap_cluster_net.excuate()
