import scipy.sparse as ss
import random
import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


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

    # sparse_mx.row/sparse_mx.col  <class 'numpy.ndarray'> [   0    0    0 ... 2687 2694 2706]
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,) [1 1 1 ... 1 1 1]
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape


def top_k(matrix, k):
    kth_largest = torch.topk(matrix, k=k, dim=1).values[:, -1:]
    mask = matrix >= kth_largest
    matrix = matrix * mask
    return matrix


def bottom_k(matrix, k):
    kth_smallest = torch.topk(matrix, k=k, dim=1, largest=False).values[:, -1:]
    mask = matrix > kth_smallest
    matrix = matrix * mask
    return matrix


def bestMap(y_pred, y_true):
    from scipy.optimize import linear_sum_assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    np.asarray(ind)
    ind = np.transpose(ind)
    label = np.zeros(y_pred.size)
    for i in range(y_pred.size):
        label[i] = ind[y_pred[i]][1]
    return label.astype(np.int64)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    indx_list = []
    for i in range(len(ind[0])):
        indx_list.append((ind[0][i], ind[1][i]))
    return sum([w[i1, j1] for (i1, j1) in indx_list]) * 1.0 / y_pred.size


def clustering_purity(labels_true, labels_pred):
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins=range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct / len(y_pred)

    return Purity


def b3_precision_recall_fscore(labels_true, labels_pred):
    # Check that labels_* are 1d arrays and have the same size
    labels_pred = bestMap(labels_pred, labels_true)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]
        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection
        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)
    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score, precision, recall


def clusteringMetrics(trueLabel, predictiveLabel):
    y_pred = bestMap(predictiveLabel, trueLabel)
    ACC = cluster_acc(trueLabel, y_pred)
    NMI = normalized_mutual_info_score(trueLabel, y_pred)
    Purity = clustering_purity(trueLabel.reshape((-1, 1)), y_pred.reshape(-1, 1))
    ARI = metrics.adjusted_rand_score(trueLabel, y_pred)
    Fscore, Precision, Recall = b3_precision_recall_fscore(trueLabel, y_pred)

    return ACC, NMI, Purity, ARI, Fscore, Precision, Recall


def KMeansClustering(features, gnd, clusterNum, randNum):
    kmeans = KMeans(n_clusters=clusterNum, n_init=1, max_iter=500,
                    random_state=randNum)
    estimator = kmeans.fit(features)
    clusters = estimator.labels_
    labels = np.zeros_like(clusters)
    for i in range(clusterNum):
        mask = (clusters == i)
        labels[mask] = mode(gnd[mask])[0]
    return labels


def StatisticClustering(features, gnd, clusterNum):
    repNum = 5
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    PurityList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    FscoreList = np.zeros((repNum, 1))
    PrecisionList = np.zeros((repNum, 1))
    RecallList = np.zeros((repNum, 1))

    for i in range(repNum):
        predictiveLabel = KMeansClustering(F.normalize(features, dim=1).cpu().numpy(), gnd, clusterNum,
                                           random.randint(1, 9999999))
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, predictiveLabel)

        ACCList[i] = ACC
        NMIList[i] = NMI
        PurityList[i] = Purity
        ARIList[i] = ARI
        FscoreList[i] = Fscore
        PrecisionList[i] = Precision
        RecallList[i] = Recall
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    Puritymean_std = np.around([np.mean(PurityList), np.std(PurityList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    Fscoremean_std = np.around([np.mean(FscoreList), np.std(FscoreList)], decimals=4)
    Precisionmean_std = np.around([np.mean(PrecisionList), np.std(PrecisionList)], decimals=4)
    Recallmean_std = np.around([np.mean(RecallList), np.std(RecallList)], decimals=4)
    return ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std


def spec_clustering(W, n_clusters, labels):
    Dn = torch.diag(torch.pow(torch.sum(W, dim=1) + 1e-8, -0.5))
    A = Dn @ W @ Dn
    L = torch.eye(W.shape[0]).to(W.device) - A
    L[torch.isnan(L)] = 0
    s, v = torch.linalg.eigh(L)
    v = v[:, :n_clusters].float()
    return StatisticClustering(v, labels, n_clusters)
