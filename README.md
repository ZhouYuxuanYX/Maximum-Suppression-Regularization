# MaxSup: Overcoming Representation Collapse in Label Smoothing

**Max Suppression (MaxSup)** is a novel regularization technique that overcomes the shortcomings of traditional **Label Smoothing (LS)**. While LS prevents overconfidence by softening one-hot labels, it inadvertently collapses intra-class feature diversity and can boost overconfident errors. In contrast, **MaxSup** applies a uniform smoothing penalty to the model’s top prediction—regardless of correctness—preserving richer per-sample information and improving both classification performance and downstream transfer.

---

## Table of Contents

1. [Overview](#overview)
2. [Methodology: MaxSup vs. Label Smoothing](#methodology-maxsup-vs-label-smoothing)
3. [Enhanced Feature Representation](#enhanced-feature-representation)
   - [Qualitative Evaluation](#qualitative-evaluation)
   - [Quantitative Evaluation](#quantitative-evaluation)
4. [Training Vision Transformers with MaxSup](#training-vision-transformers-with-maxsup)
   - [Accelerated Data Loading via Caching (Optional)](#accelerated-data-loading-via-caching-optional)
   - [Preparing Data and Annotations for Caching](#preparing-data-and-annotations-for-caching)
5. [Pretrained Weights](#pretrained-weights)
6. [Training ConvNets with MaxSup](#training-convnets-with-maxsup)
7. [Logit Characteristic Visualization](#logit-characteristic-visualization)
8. [Citation](#citation)
9. [References](#references)

---

## Overview

Traditional Label Smoothing (LS) replaces one-hot labels with a smoothed version to reduce overconfidence. However, LS can over-tighten feature clusters within each class and may reinforce errors by making mispredictions overconfident. **MaxSup** tackles these issues by applying a smoothing penalty to the model's **top-1 logit** output regardless of whether the prediction is correct, thus preserving intra-class diversity and enhancing inter-class separation. The result is improved performance on both classification tasks and downstream applications such as linear transfer and image segmentation.

---

## Methodology: MaxSup vs. Label Smoothing

Label Smoothing softens the target distribution by blending the one-hot vector with a uniform distribution. Although effective at reducing overconfidence, LS inadvertently introduces two effects:
- A **regularization term** that limits the sharpness of predictions.
- An **error-enhancement term** that can cause overconfident wrong predictions.

**MaxSup** addresses this by uniformly penalizing the highest logit output, whether it corresponds to the true class or not. This approach enforces a consistent regularization effect across all samples. In formula form:

```math
L_{\text{MaxSup}} = \alpha \left( z_{\max} - \frac{1}{K}\sum_{k=1}^{K} z_k \right),
```

where \( z_{\max} \) is the highest logit among the \( K \) classes. This mechanism prevents the prediction distribution from becoming too peaky while preserving informative signals from non-target classes.

---

## Enhanced Feature Representation

### Qualitative Evaluation

MaxSup-trained models display richer intra-class feature diversity compared to models trained with traditional LS. Feature embedding visualizations show that while LS forces features into tight clusters, MaxSup preserves finer-grained differences among samples. Grad-CAM analyses also demonstrate that MaxSup-trained models focus more precisely on relevant class-discriminative regions.

![Improved Feature Representation](Improved_Feature.png)  
**Figure 1:** Feature representations. MaxSup maintains greater intra-class diversity and clear inter-class boundaries.

![Grad-CAM Analysis](gradcam.png)  
**Figure 2:** Grad-CAM visualizations. The MaxSup model (row 2) accurately highlights target objects, whereas the LS model (row 3) and Baseline (row 4) show more diffuse activations.

### Quantitative Evaluation

We evaluated feature representations on ResNet-50 trained on ImageNet-1K. Intra-class variation (reflecting the diversity within classes) and inter-class separability (indicating class distinctiveness) were measured. Additionally, a linear transfer learning task on CIFAR-10 was performed.

**Table 1: Feature Representation Metrics (ResNet-50 on ImageNet-1K)**

| Method                    | Intra-class Var. (Train) | Intra-class Var. (Val) | Inter-class Sep. (Train) | Inter-class Sep. (Val) |
|---------------------------|--------------------------|------------------------|--------------------------|------------------------|
| **Baseline**              | 0.3114                   | 0.3313                 | 0.4025                   | 0.4451                 |
| **Label Smoothing**       | 0.2632                   | 0.2543                 | 0.4690                   | 0.4611                 |
| **Online LS**             | 0.2707                   | 0.2820                 | 0.5943                   | 0.5708                 |
| **Zipf’s LS**             | 0.2611                   | 0.2932                 | 0.5522                   | 0.4790                 |
| **MaxSup (ours)**         | **0.2926**               | **0.2998**             | 0.5188                   | 0.4972                 |

*Higher intra-class variation indicates more preserved sample-specific details, while higher inter-class separability suggests better class discrimination.*

**Table 2: Linear Transfer Accuracy on CIFAR-10**

| Pretraining Method   | Accuracy (%) |
|----------------------|--------------|
| **Baseline**         | 81.43        |
| **Label Smoothing**  | 74.58        |
| **MaxSup**           | **81.02**    |

Label Smoothing degrades transfer accuracy due to its over-smoothing effect, whereas MaxSup nearly matches the baseline performance while still offering improved calibration.

---

## Training Vision Transformers with MaxSup

We integrated MaxSup into the training pipeline for Vision Transformers using the [DeiT](https://github.com/facebookresearch/deit) framework.

### To Train a ViT with MaxSup:

```bash
cd Deit
python train_with_MaxSup.sh
```

This script trains a DeiT-Small model on ImageNet-1K with MaxSup regularization.

### Accelerated Data Loading via Caching (Optional)

For improved data loading efficiency on systems with slow I/O, a caching mechanism is provided. This feature compresses the ImageNet dataset into ZIP files and loads them into memory. Enable caching by adding the `--cache` flag to the training script.

### Preparing Data and Annotations for Caching

1. **Create ZIP Archives:**  
   In your ImageNet data directory, run:
   ```bash
   cd data/ImageNet
   zip -r train.zip train
   zip -r val.zip val
   ```

2. **Mapping Files:**  
   Download `train_map.txt` and `val_map.txt` from our release assets and place them in the `data/ImageNet` directory. The directory should appear as follows:
   ```
   data/ImageNet/
   ├── train_map.txt      # Relative paths and labels for training images
   ├── val_map.txt        # Relative paths and labels for validation images
   ├── train.zip          # Compressed training images
   └── val.zip            # Compressed validation images
   ```
   - **train_map.txt:** Each line should be in the format `<class_folder>/<image_filename>\t<label>`.
   - **val_map.txt:** Each line should be in the format `<image_filename>\t<label>`.

---

## Pretrained Weights

Pretrained checkpoints for both Vision Transformers and ConvNets are available:

- **Vision Transformer (DeiT-Small):** Available in the release titled **"checkpoint_deit"**. This includes the model `.pth` file and training logs.
- **ConvNet (ResNet-50):** Pretrained weights can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1loCatg32vXeUHby2GH6wjtAZjz3T90YJ/view?usp=sharing). Baseline and LS variants are also provided for comparison.

These checkpoints can be used for direct evaluation or fine-tuning on downstream tasks.

---

## Training ConvNets with MaxSup

The `Conv/` directory provides scripts for training convolutional networks with MaxSup:

- **Conv/ffcv:** Contains scripts to reproduce ImageNet results using FFCV for efficient data loading. See `Conv/ffcv/README.md` for details.
- **Conv/common_resnet:** Contains additional experiments with ResNet architectures. Refer to `Conv/common_resnet/README.md` for further instructions.

---

## Logit Characteristic Visualization

The `viz/` directory contains a toolkit to analyze the distribution of logits produced by models trained with LS versus MaxSup.

### Step 1: Extract Logits

Run the following command to extract logits from your trained model:

```bash
python viz/logits.py \
    --checkpoint /path/to/model_checkpoint.pth \
    --output /path/to/save/logits_labels.pt
```

- `--checkpoint`: Path to your model checkpoint.
- `--output`: Destination file for the extracted logits and labels.

### Step 2: Analyze Logits

After extraction, run:

```bash
python viz/analysis.py --input /path/to/save/logits_labels.pt --output /path/to/analysis_results/
```

This script generates:
- A histogram of near-zero logit proportions.
- A scatter plot comparing top-1 probabilities with near-zero proportions.
- Saved visualizations for side-by-side comparisons.

![Logit Visualization](logit.png)  
**Figure 3:** Logit distribution comparing LS and MaxSup. 

## References
- **DeiT (Vision Transformer):**  
  Touvron et al., *Training Data-Efficient Image Transformers & Distillation through Attention*, ICML 2021. [GitHub](https://github.com/facebookresearch/deit).
- **Grad-CAM:**  
  Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- **Online Label Smoothing:** See paper for details.
- **Zipf’s Label Smoothing:** See paper for details.

---

This repository provides the official implementation of MaxSup. Contributions and discussions are welcome. For any questions or issues, please open an issue on GitHub or contact the authors directly.

---
