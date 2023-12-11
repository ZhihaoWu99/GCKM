import torch
import random
import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize, minmax_scale, robust_scale
from sklearn.model_selection import StratifiedKFold


def load_data(args):
    data = sio.loadmat(args.path + '/node level/' + args.dataset + '.mat')
    features = data['X']
    if ss.isspmatrix(features):
        features = features.todense()
    features = normalize(features)
    adj = data['adj']
    if ss.isspmatrix(adj):
        adj = adj.todense()

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    n_class = len(np.unique(labels))
    print('dataset: {}\n# node: {}\n# class: {}'.format(
        args.dataset, features.shape[0], n_class
    ))
    if args.rand_split:
        idx_train, idx_test, idx_val, idx_unlabeled = generate_partition_random(data, 20)
    else:
        idx_train, idx_test, idx_val, idx_unlabeled = generate_partition(data)
    adj_hat = torch.from_numpy(construct_adj_hat(adj).todense()).float().to(args.device)
    features = torch.from_numpy(features).float().to(args.device)
    labels = torch.from_numpy(labels).long()
    return adj_hat, features, labels, n_class, idx_train, idx_val, idx_test


def construct_adj_hat(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = ss.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = 2 * ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_hat = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_hat


def generate_partition(data):
    idx_train = data['train'].flatten()
    idx_val = data['val'].flatten()
    idx_test = data['test'].flatten()
    print('train: {}, val: {}, test: {}'.format(len(idx_train), len(idx_val), len(idx_test)))
    return idx_train, idx_test, idx_val, []


def generate_partition_random(labels, num_perclass):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}
    for label in each_class_num.keys():
        labeled_each_class_num[label] = num_perclass
    num_test = 1000
    num_val = 500
    idx_train = []
    idx_test = []
    idx_val = []
    idx_unlabeled = []
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            idx_train.append(index[idx])
        else:
            idx_unlabeled.append(index[idx])
            if num_test > 0:
                num_test -= 1
                idx_test.append(index[idx])
            elif num_val > 0:
                num_val -= 1
                idx_val.append(index[idx])
    print('train: {}, val: {}, test: {}'.format(len(idx_train), len(idx_val), len(idx_test)))
    return idx_train, idx_test, idx_val, idx_unlabeled


def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = ss.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0


def load_graph_data(args):
    print('dataset: {}'.format(args.dataset))
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open((args.path + 'graph level/%s/%s.txt') % (args.dataset, args.dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if args.degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print("# graph: %d" % len(g_list))
    print('# class: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    return g_list, len(label_dict)


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    return train_graph_list, test_graph_list


def separate_data_idx(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx
