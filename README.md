Graph Convolutional Kernel Machine
====
This is the implementation of GCKM proposed in our paper:

[Zhihao Wu](https://zhihaowu99.github.io/), [Zhao Zhang](http://faculty.hfut.edu.cn/cszzhang/zh_CN/), and [Jicong Fan](https://jicongfan.github.io/)*. [Graph Convolutional Kernel Machine versus Graph Convolutional Networks](https://openreview.net/forum?id=SFfOt1oDsX), NeurIPS 2023.



![framework](./Framework.jpg)

## Requirement

  * Python = 3.9
  * PyTorch = 1.11
  * Numpy = 1.21
  * Scikit-learn = 1.1
  * Scipy = 1.8
  * Networkx = 2.8
  * Tqdm = 4.64

## Quick Start
Unzip the dataset files
```
unzip ./data/datasets.7z
```
For node classification task, run 
```
python node_classification.py --dataset Cora
```
For node clustering task, run 
```
python node_clustering.py --dataset Cora
```
For graph classification task, run
```
python graph_classification.py --dataset MUTAG
```

Note that the default parameters may not be the best to reproduce our results in the paper.

## Tuning
For datasets that are not included in our paper, please run
```
python para_tuning.py --dataset *new_dataset_name*
```
to search for the best parameters on validation set. 

Running ```para_tuning.py``` requires extra package
  * [Hyperopt](https://github.com/hyperopt/hyperopt)


## Dataset

### Node-level
  * Cora
  * Citeseer
  * Pubmed
  * ACM
  * Actor
  * Chameleon
  * CoraFull
  * Squirrel
  * UAI
  * OGB-Arxiv*

*Please feel free to contact me via zhihaowu1999@gmail.com for codes regarding OGB-Arxiv (due to the large file).

### Graph-level
  * COLLAB
  * IMDBBINARY
  * IMDBMULTI
  * MUTAG
  * PROTEINS
  * PTC

Saved in ```./data/node level.7z``` and ```./data/graph level.7z```

*Note: Please unzip the datasets folders first; Random data splitting function can be found in Dataloader.py.*

## Reference
```
@inproceedings{wu2023graph,
  title={Graph Convolutional Kernel Machine versus Graph Convolutional Networks},
  author={Wu, Zhihao and Zhang, Zhao and Fan, Jicong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
