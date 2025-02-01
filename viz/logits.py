
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Tuple
import warnings
import json
from train_imagenet import ImageNetTrainer, make_config, get_current_config
import argparse
from tqdm import tqdm
from torchvision import models


from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        # Create filter on same device as conv layer
        default_filter = torch.tensor(
            [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], 
            device=conv.weight.device,  # Match device with conv layer
            dtype=conv.weight.dtype      # Match precision with conv layer
        ) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = torch.nn.functional.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)


def analyze_logits(logits: torch.Tensor, threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze logit values and calculate proportion near zero for each sample.
    
    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, num_classes)
        threshold (float): Absolute threshold for considering logit as "near zero"
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - proportions: Array of proportions of near-zero logits per sample
            - top_probs: Array of top-1 probabilities for each sample
    """
    with torch.no_grad():
        # Calculate proportion of logits near zero for each sample
        near_zero = (torch.abs(logits) < threshold).float()
        proportions = near_zero.mean(dim=1).cpu().numpy()
        
        # Calculate top-1 probabilities
        probs = torch.softmax(logits, dim=1)
        top_probs, _ = probs.max(dim=1)
        top_probs = top_probs.cpu().numpy()
    
    return proportions, top_probs

def plot_logit_analysis(proportions: np.ndarray, 
                        top_probs: np.ndarray,
                        save_dir: Path,
                        threshold: float = 0.01,
                        num_bins: int = 50) -> None:
    """
    Visualize logit distribution characteristics with histograms and scatter plots.
    
    Args:
        proportions (np.ndarray): Proportions of near-zero logits per sample
        top_probs (np.ndarray): Top-1 probabilities for each sample
        save_dir (Path): Directory to save visualization figures
        threshold (float): Threshold used for near-zero determination
        num_bins (int): Number of bins for histograms
    """
    
    
    plt.figure(figsize=(12, 5))
    
    # Histogram of near-zero proportions
    plt.subplot(1, 2, 1)
    plt.hist(proportions, bins=num_bins, edgecolor='black')
    plt.title(f'Proportion of Logits < ±{threshold}')
    plt.xlabel('Proportion of Near-Zero Logits')
    plt.ylabel('Count')
    
    # Scatter plot of top prob vs near-zero proportion
    plt.subplot(1, 2, 2)
    plt.scatter(top_probs, proportions, alpha=0.6)
    plt.title('Top Probability vs Near-Zero Logits')
    plt.xlabel('Top-1 Probability')
    plt.ylabel('Proportion of Near-Zero Logits')
    plt.grid(True)
    
    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'logit_analysis.png')


class LogitExtractor:
    """Class for loading trained model and extracting logits from validation set"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        """
        Args:
            checkpoint_path (str): Path to saved model weights
            device (str): Device to load model on
        """
        self.device = torch.device(device)
        self.model = self.load_model(checkpoint_path)
        # Convert model to half precision except for BatchNorm layers
        #self.model = self.model.half()
        #for module in self.model.modules():
        #    if isinstance(module, torch.nn.BatchNorm2d):
        #        module.float()
        self.model.eval()
        
    def load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load trained model with architecture from original training setup"""
        # Load original training config
        config_path = Path(checkpoint_path).parent / 'params.json'
        with open(config_path) as f:
            params = json.load(f)
        
        # Recreate original model architecture
        arch = params['model.arch']
        model = getattr(models, arch)(pretrained=False)
        def apply_blurpool(mod: torch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if params['training.use_blurpool']: apply_blurpool(model)
        model = model.to(self.device)
        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Remove DDP wrapper prefix if present
        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                state_dict[k] = v
                
        model.load_state_dict(state_dict)
        return model
    
    @torch.no_grad()
    def get_logits(self, val_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract logits and labels from validation set
        
        Args:
            val_loader (DataLoader): Validation set loader
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, labels) tensors
        """
        all_logits, all_labels = [], []
        
        for images, labels in tqdm(val_loader):
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
        return torch.cat(all_logits), torch.cat(all_labels)

def create_val_loader(val_dataset, num_workers, batch_size,
                        resolution, distributed):
    this_device = f'cuda:0'
    val_path = Path(val_dataset)
    assert val_path.is_file()
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device),
        non_blocking=True)
    ]

    loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)
    return loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract validation set logits')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for logits (.pt file)')
    args = parser.parse_args()

    # Load config from checkpoint directory
    config_path = Path(args.checkpoint).parent / 'params.json'
    with open(config_path) as f:
        params = json.load(f)

    # Create validation loader using original training parameters
    val_loader = create_val_loader(
        val_dataset=params['data.val_dataset'],
        num_workers=int(params['data.num_workers']),
        batch_size=int(params['validation.batch_size']),
        resolution=int(params['validation.resolution']),
        distributed=False
    )

    # Initialize extractor and get logits
    extractor = LogitExtractor(args.checkpoint)
    logits, labels = extractor.get_logits(val_loader)
    
    # Save results
    torch.save({'logits': logits, 'labels': labels}, args.output)
    print(f"Saved logits and labels to {args.output}")
