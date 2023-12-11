import torch
import configparser
import ast
from Dataloader import load_graph_data
from args import parameter_parser
from main import graph_classification


if __name__ == '__main__':
    args = parameter_parser()
    args.device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    if args.config:
        conf = configparser.ConfigParser()
        config_path = './config/config_graph_classification.ini'
        conf.read(config_path)
        assert conf.has_section(args.dataset)
        args.gamma = ast.literal_eval(conf.get(args.dataset, 'gamma'))
        args.pow = conf.getint(args.dataset, 'pow')
        args.C = conf.getfloat(args.dataset, 'C')
        args.degree_as_tag = conf.getboolean(args.dataset, 'degree_as_tag')
    # Load data
    graphs, n_class = load_graph_data(args)
    graph_classification(args, graphs)




