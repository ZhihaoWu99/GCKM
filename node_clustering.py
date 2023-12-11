import random
import numpy as np
import torch
import configparser
import ast
from Dataloader import load_data
from args import parameter_parser
from main import kernel, clustering


if __name__ == '__main__':
    args = parameter_parser()
    args.device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if args.config:
        conf = configparser.ConfigParser()
        config_path = './config/config_node_clustering.ini'
        conf.read(config_path)
        assert conf.has_section(args.dataset)
        args.gamma = ast.literal_eval(conf.get(args.dataset, 'gamma'))
        args.dim = ast.literal_eval(conf.get(args.dataset, 'dim'))
        args.pow = conf.getint(args.dataset, 'pow')
        args.k = conf.getint(args.dataset, 'k')

    # Load data
    adj, features, labels, n_class, idx_train, idx_val, idx_test = load_data(args)
    args.dim = [features.shape[1]] + args.dim
    K = kernel(args, adj, features)
    clustering(args, K, labels, n_class)
