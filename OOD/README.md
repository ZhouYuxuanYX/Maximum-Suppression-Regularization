## Experiments on OOD dataset: CIFAR10-C (Maxsup vs Label Smoothing)

We follow the setup from [Robust Classification by Coupling Data Mollification with Label Smoothing](https://arxiv.org/abs/2406.01494)


### Quick Start

- Download CIFAR10 (see `/data`), and run `data/process-datasets.py`. The code `src/data.py` expects datasets to be .npy files with (N,C,W,H) array inside


Train MaxSup on CIFAR10-C using Resnet50 as backbone:

- `python3 train.py -d cifar10 -n resnet50 --loss maxsup`   

Train Label Smoothing (LS) on CIFAR10-C using Resnet50 as backbone:

- `python3 train.py -d cifar10 -n resnet50 --loss lsce` 

### Results

| Metric         | MaxSup | LS |
|----------------|--------|------------------|
| Error (Corr)   | **0.3951** | **0.3951**           |
| NLL (Corr)     | 1.8431 | **1.5730**           |
| ECE (Corr)     | **0.1479** | 0.1741           |

