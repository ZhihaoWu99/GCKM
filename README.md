Graph Convolutional Kernel Machine
====
This is the implementation of GCKM proposed in our paper:

[Zhihao Wu](https://zhihaowu99.github.io/), [Zhao Zhang](http://faculty.hfut.edu.cn/cszzhang/zh_CN/), and [Jicong Fan](https://jicongfan.github.io/)*. [Graph Convolutional Kernel Machine versus Graph Convolutional Networks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html), NeurIPS 2023.

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
to search for the best parameters on the validation set. 

Running ```para_tuning.py``` requires
  * [Hyperopt](https://github.com/hyperopt/hyperopt)


## Dataset
Please unzip the datasets folders saved in ```./data/node level.7z``` and ```./data/graph level.7z``` first.

```
data/
│
├── node level/
│   ├── ACM.mat
│   ├── Actor.mat
│   ├── Chameleon.mat
│   ├── Citeseer.mat
│   ├── Cora.mat
│   ├── CoraFull.mat
│   ├── Pubmed.mat
│   ├── Squirrel.mat
│   └── UAI.mat
│
└── graph level/
    ├── COLLAB
    ├── IMDBBINARY
    ├── IMDBMULTI
    ├── MUTAG
    ├── PROTEINS
    └── PTC
```

*Feel free to contact me via zhihaowu1999@gmail.com for codes regarding OGB-Arxiv (due to the large file). Random data splitting function can be found in Dataloader.py.*

## Reference
```
@inproceedings{wu2023graph,
  title={Graph Convolutional Kernel Machine versus Graph Convolutional Networks},
  author={Wu, Zhihao and Zhang, Zhao and Fan, Jicong},
  booktitle={Advances in Neural Information Processing Systems},
  pages = {19650--19672},
  volume = {36},
  year={2023}
}
```
