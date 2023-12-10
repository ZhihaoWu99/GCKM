from __future__ import division
from __future__ import print_function
import random
import numpy as np
import torch
import configparser
import ast
from Dataloader import load_data, generate_partition
from args import parameter_parser
from main import kernel, classification
from PCA import pca, gpca


if __name__ == '__main__':
    args = parameter_parser()
    args.device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    if args.config:
        conf = configparser.ConfigParser()
        config_path = './config/config_node_classification.ini'
        conf.read(config_path)
        args.gamma = ast.literal_eval(conf.get(args.dataset, 'gamma'))
        args.dim = ast.literal_eval(conf.get(args.dataset, 'dim'))
        args.pow = conf.getint(args.dataset, 'pow')
        args.C = conf.getfloat(args.dataset, 'C')

    # Load data
    adj, features, labels, n_class, idx_train, idx_val, idx_test = load_data(args)
    args.dim = [features.shape[1]] + args.dim
    K = kernel(args, adj, features)
    # classification(args, K, labels, idx_train, idx_val, idx_test)
    pca(features.cpu(), K.cpu(), labels)
    gpca(features, adj, labels, idx_test, 0.2)
