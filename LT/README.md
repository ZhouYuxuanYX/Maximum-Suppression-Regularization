## Long-Tailed image Classification (Maxsup vs Label Smoothing vs Focal Loss)

This experiment is built on top of [Decoupling](https://github.com/facebookresearch/classifier-balancing) and [LVIS](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/). The main body of **[the proposed Causal-TDE](https://kaihuatang.github.io/Files/long-tail.pdf)** is under [./models/CausalNormClassifier.py](models/CausalNormClassifier.py) and [run_networks.py](run_networks.py) 


### Training

For Long-Tailed CIFAR-10 dataset with imbalance ratio 50, using ResNet32 as backbone and Maxsup as loss:

```bash
python main.py --cfg ./config/CIFAR10_LT/causal_norm_32_maxsup_imb50.yaml
```


If you want to change the imbalance ratio and loss type, you can change '--cfg' to different config files in [config](config/CIFAR10_LT/).

### Testing

For Long-Tailed CIFAR-10 using ResNet32:
```bash
python main.py --cfg ./config/CIFAR10_LT/causal_norm_32_maxsup_imb50.yaml --test --model_dir ./logs/CIFAR10_LT/models/resnet32_e200_warmup_causal_norm_ratio50_maxsup/latest_model_checkpoint.pth
```


### Results


| **Dataset**           | **Split** | **Imbalance Ratio** | **Backbone** | **Method**      | **Overall** | **Many** | **Medium** | **Low** |
|-----------------------|-----------|----------------------|--------------|------------------|------------|--------|----------|--------|
| Long-tailed CIFAR-10  | val       | 50                   | Resnet32     | Focal Loss       | 77.4       |  76.0      |  89.7        |   0.0     |
|                       |           |                      |              | LS              | 81.2       |  81.6      |   77.0       |   0.0     |
|                       |           |                      |              | MaxSup          | **82.1**   |  82.5      |     78.1     |   0.0     |
| Long-tailed CIFAR-10  | test      | 50                   | Resnet32     | Focal Loss       | 76.8       |  75.3      |  90.4        |   0.0     |
|                       |           |                      |              | LS              | 80.5       |  81.1      |   75.4       |   0.0     |
|                       |           |                      |              | MaxSup          | **81.4**   |  82.3      |  73.4        |   0.0     |
| Long-tailed CIFAR-10  | val       | 100                  | Resnet32     | Focal Loss       | 75.1       |  71.8      |  88.3        |   0.0     |
|                       |           |                      |              | LS              | 76.6       |   80.6     |   60.7       |  0.0      |
|                       |           |                      |              | MaxSup          | **77.1**   |   80.1     |  65.1        |   0.0     |
| Long-tailed CIFAR-10  | test      | 100                  | Resnet32     | Focal Loss       | 74.7       |  71.6      |     87.2     |     0.0   |
|                       |           |                      |              | LS              | 76.4       |   80.8     |    59.0      |   0.0     |
|                       |           |                      |              | MaxSup          | **76.4**   |  79.9      |    62.4      |    0.0    |

