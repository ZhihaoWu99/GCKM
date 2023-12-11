import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu.")
    parser.add_argument("--path", type=str, default="./data/", help="Path of datasets.")
    parser.add_argument("--dataset", type=str, default="Citeseer", help="Name of datasets.")
    parser.add_argument("--config", action='store_true', default=True, help="Read configuration file.")

    parser.add_argument("--rff", type=bool, default=False, help="Use GCKM-E.")
    parser.add_argument('--dim', nargs='+', type=int, default=[2000, 2000])
    parser.add_argument('--pow', type=int, default=1, help="power of adj.")
    parser.add_argument("--n_label", type=int, default=20, help="Number of labeled samples per class.")
    parser.add_argument('--gamma', nargs='+', type=float, default=[1, 1])

    # Parameter only for classification
    parser.add_argument('--C', type=float, default=1, help="Regularization parameter of SVM.")
    parser.add_argument('--rand_split', type=bool, default=False, help="Random split.")
    # Parameter only for graph classification
    parser.add_argument('--degree_as_tag', action="store_true", default=False, help='Degree as node features.')
    # Parameter only for node clustering
    parser.add_argument("--k", type=int, default=-1, help="Keep the k largest value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fix_seed", action='store_true', default=False, help="fix the seed.")

    args = parser.parse_args()

    return args
