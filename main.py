from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from sklearn.svm import SVC
from model import GCKM, GCKM_E
from utils import top_k, spec_clustering
from Dataloader import separate_data_idx


def kernel(args, adj, features):
    adj_q = torch.matrix_power(adj, args.pow)
    if args.rff:
        print("model: GCKM-E")
        model = GCKM_E(args.gamma, args.dim, 1).to(args.device)
    else:
        print("model: GCKM")
        model = GCKM(args.gamma).to(args.device)
    K = model(adj_q, features)
    return K


def classification(args, K, labels, idx_train, idx_val, idx_test):
    K_train = K[np.ix_(idx_train, idx_train)]
    K_test = K[np.ix_(idx_test, idx_train)]
    K_val = K[np.ix_(idx_val, idx_train)]
    svm = SVC(kernel='precomputed', probability=True, C=args.C, tol=1e-5, max_iter=2000).fit(K_train.to('cpu').numpy(), labels[idx_train].numpy())
    y_pred_val = svm.predict(K_val.to('cpu').numpy())
    y_pred_test = svm.predict(K_test.to('cpu').numpy())
    acc_val = np.sum(y_pred_val == labels[idx_val].numpy())/y_pred_val.size
    acc_test = np.sum(y_pred_test == labels[idx_test].numpy())/y_pred_test.size
    print("gamma: {}, pow: {}, C: {}\nACC_val: {:.2f}, ACC_test: {:.2f}".format(args.gamma, args.pow, args.C, acc_val*100, acc_test*100))


def clustering(args, K, labels, n_class):
    K_diag = torch.diag(torch.diag(K))
    K = K - K_diag
    if args.k != -1:
        K_mask = top_k(K, args.k)
        K = (K_mask + K_mask.t()) / 2
    results = spec_clustering(K, n_class, labels.cpu().numpy())
    print("gamma: {}, pow: {}\nACC: {:.2f} ({:.2f}), NMI: {:.2f} ({:.2f}), ARI: {:.2f} ({:.2f}), F: {:.2f} ({:.2f})"
          .format(args.gamma, args.pow,
                  results[0][0] * 100, results[0][1] * 100, results[1][0] * 100, results[1][1] * 100,
                  results[3][0] * 100, results[3][1] * 100, results[4][0] * 100, results[4][1] * 100))


def sys_normalized_adjacency(adj):
    row_sum = adj.sum(1)
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = row_sum ** (-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def kernel_rbf(X, Y, gamma):
    ni = X.shape[0]
    nj = Y.shape[0]
    Si = torch.unsqueeze(torch.diag(torch.matmul(X, X.T)), 0).T @ torch.ones(1, nj)
    Sj = torch.ones(ni, 1) @ torch.unsqueeze(torch.diag(torch.matmul(Y, Y.T)), 0)
    Sij = torch.matmul(X, Y.T)
    D = Si + Sj - 2 * Sij
    K = torch.exp(-D * gamma)
    return K


def graph_classification(args, graphs):
    N = len(graphs)
    power_A = args.pow
    gm1 = args.gamma[0]
    gm2 = args.gamma[1]
    graph_sizes = []
    for i in range(N):
        ni, d = graphs[i].node_features.shape
        graph_sizes = np.append(graph_sizes, ni)
    X = []
    A = []
    S = []
    acc_test = []

    for i in range(N):
        Ai = torch.sparse.FloatTensor(graphs[i].edge_mat, torch.ones(graphs[i].edge_mat.shape[1])).to_dense()
        Ai = Ai + torch.eye(Ai.shape[0])
        Ai = torch.matrix_power(sys_normalized_adjacency(Ai), power_A)
        Xi = torch.matmul(Ai, graphs[i].node_features)
        Si = Ai @ kernel_rbf(Xi, Xi, gm1) @ Ai.T
        X.append(Xi)
        A.append(Ai)
        S.append(Si)
    K = torch.zeros(N, N)
    for i in range(N):
        for j in range(i + 1, N):
            ni = S[i].shape[0]
            nj = S[j].shape[0]
            Si = torch.unsqueeze(torch.diag(S[i]), 0).T @ torch.ones(1, nj)
            Sj = torch.ones(ni, 1) @ torch.unsqueeze(torch.diag(S[j]), 0)
            Sij = A[i] @ kernel_rbf(X[i], X[j], gm1) @ A[j].T
            Dij = Si + Sj - 2 * Sij
            K[i, j] = torch.ones(1, ni) @ torch.exp(-Dij * gm2) @ torch.ones(nj, 1)
    K = (K + K.T) + torch.eye(N)
    for t in range(10):
        # 10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
        id_train, id_test = separate_data_idx(graphs, 0, t)  # random seed / fold_idx
        y_train = torch.LongTensor([graphs[i].label for i in id_train]).numpy()
        y_test = torch.LongTensor([graphs[i].label for i in id_test]).numpy()
        K_train = K[:, id_train]
        K_train = K_train[id_train, :]
        K_test = K[:, id_test]
        K_test = K_test[id_train, :]
        svm = SVC(kernel='precomputed', C=args.C, tol=1e-3, max_iter=2000).fit(K_train.to('cpu').numpy(), y_train)
        y_pred = svm.predict(K_test.to('cpu').numpy().T)
        acc_test.append((np.sum(y_pred == y_test) / y_pred.size) * 100)
    print("gamma: {}, pow: {}\nACC: {:.2f} ({:.2f})"
          .format(args.gamma, args.pow, np.mean(acc_test, axis=0), np.std(acc_test, axis=0)))


