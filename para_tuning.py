from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import torch
import configparser
import numpy as np
from Dataloader import load_data
from sklearn.svm import SVC
from args import parameter_parser
from hyperopt import fmin, hp, tpe, STATUS_OK
from main import kernel


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
    return acc_val, acc_test


def GCKM_obj(space):
    # Round the parameters
    args.gamma = [round(space['gamma1'], 2), round(space['gamma2'], 2)]
    args.pow = int(space['power_A'])
    args.C = round(space['C'], 2)

    # Perform GCKM
    K = kernel(args, adj, features)
    acc_val, acc_test = classification(args, K, labels, idx_train, idx_val, idx_test)
    torch.cuda.empty_cache()
    return {'loss': -acc_val, 'acc_test': acc_test, 'status': STATUS_OK}


if __name__ == '__main__':
    args = parameter_parser()
    args.device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    adj, features, labels, n_class, idx_train, idx_val, idx_test = load_data(args)
    args.dim = [features.shape[1]] + args.dim

    # Set the searching parameters
    evals = 2000
    space = {'power_A': hp.uniform('power_A', 1, 10),
             'gamma1': hp.uniform('gamma1', 0.001, 10),
             'gamma2': hp.uniform('gamma2', 0.001, 10),
             'C': hp.uniform('C', 0.01, 10),
             }

    # Start searching
    best = fmin(GCKM_obj, space=space, algo=tpe.suggest, max_evals=evals)
    print("===========Tuning Completed===========")
    best_result = GCKM_obj(best)
    print('Config: gamma: {}, pow: {}, C: {}'.format
          ([round(best['gamma1'], 2), round(best['gamma2'], 2)], int(best['power_A']), round(best['C'], 2)))

    config = configparser.ConfigParser()
    config_path = './config/config_node_classification.ini'
    config.read(config_path)
    assert not config.has_section(args.dataset)
    config.add_section(args.dataset)
    config.set(args.dataset, 'gamma', str([round(best['gamma1'], 2), round(best['gamma2'], 2)]))
    config.set(args.dataset, 'dim', str(args.dim[1:]))
    config.set(args.dataset, 'pow', str(int(best['power_A'])))
    config.set(args.dataset, 'C', str(round(best['C'], 2)))
    with open('./config/config_node_classification.ini', 'w') as configfile:
        config.write(configfile)






