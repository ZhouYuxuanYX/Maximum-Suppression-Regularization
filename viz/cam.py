from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm 

import torch


import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import gridspec

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_cam_method(method, model, target_layers, reshape_transform):
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }

    if method not in methods:
        raise ValueError(f"Method should be one of {list(methods.keys())}")

    if method == "ablationcam":
        return methods[method](model=model, target_layers=target_layers, reshape_transform=reshape_transform,
                               ablation_layer=AblationLayerVit())
    else:
        return methods[method](model=model, target_layers=target_layers, reshape_transform=reshape_transform)


def preprocess_input_image(image_path):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return rgb_img, input_tensor


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_cam_image(model, method, image_path, target = None, use_cuda=False, aug_smooth=False, eigen_smooth=False, fake_target=None):

    # Set the target layers
    target_layers = [model.blocks[-1].norm1]

    # Get the CAM method
    cam = get_cam_method(method, model, target_layers, reshape_transform)

    # Preprocess the input image
    rgb_img, input_tensor = preprocess_input_image(image_path)

    # get top 3 probability
    ##top3_prob, top3_indices = get_top3_probability(model, input_tensor)
    #the_rank, the_probability = get_the_rank(model, input_tensor, fake_target if fake_target else target)
    # Configure CAM parameters
    cam.batch_size = 128
    
    # Set the targets
    if not target:
        targets = None
    else:
        targets = [ClassifierOutputTarget(target)]

    # Generate the CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)

    # Take the first image in the batch
    grayscale_cam = grayscale_cam[0, :]

    # Overlay the CAM on the original image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

   #return cam_image, top3_prob, top3_indices, the_rank, the_probability
    return cam_image

def get_top3_probability(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top3_prob, top3_indices = torch.topk(probabilities, k=3)
        return top3_prob, top3_indices
    
def get_the_rank(model, input_tensor, gt):
    gt = torch.tensor([gt])
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        rank = torch.argsort(probabilities, dim=1, descending=True)
        gt_rank = rank.index_select(1, gt).item()
        gt_probability = probabilities.index_select(1, gt).item()
        return gt_rank, gt_probability



def show_cam_comparison(ours_cam: np.ndarray, pretrain_cam: np.ndarray, filename: str = None) -> None:
    """
    Display a comparison between "our CAM" and "pretrained CAM".

    Args:
        ours_cam (numpy.ndarray): Our CAM image (BGR format).
        pretrain_cam (numpy.ndarray): Pretrained CAM image (BGR format).
        filename (str, optional): If provided, save the image to the specified path.
    """
    # Convert BGR to RGB
    ours_rgb = cv2.cvtColor(ours_cam, cv2.COLOR_BGR2RGB)
    pretrain_rgb = cv2.cvtColor(pretrain_cam, cv2.COLOR_BGR2RGB)

    # Create figure and GridSpec
    fig = plt.figure(figsize=(8, 4), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

    # Define unified color mapping and range
    cmap = 'jet'
    vmin = min(np.min(ours_rgb), np.min(pretrain_rgb))
    vmax = max(np.max(ours_rgb), np.max(pretrain_rgb))

    # Plot our CAM
    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(ours_rgb, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.axis('off')

    # Plot pretrained CAM
    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(pretrain_rgb, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.axis('off')


    # Optional: Add overall title
    # fig.suptitle('CAM Comparison', fontsize=16)

    # Save or display the image
    plt.tight_layout()
    plt.show()
    plt.close()
