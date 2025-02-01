from typing import List
import numpy as np
from pathlib import Path
import lightning as L
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, ToDevice
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
import torch




IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

class DataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def train_dataloader(self):
        train_loader, _ = self.create_train_loader(
            train_dataset=self.config['train_dataset'],
            num_workers=self.config['num_workers'],
            batch_size=self.config['batch_size'],
            resolution=self.config['train_resolution'],
            distributed=True,  # TODO: Make this configurable
            in_memory=self.config['in_memory']
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = self.create_val_loader(
            val_dataset=self.config['val_dataset'],
            num_workers=self.config['num_workers'],
            batch_size=self.config['batch_size'],
            resolution=self.config['val_resolution'],
            distributed=True  # TODO: Make this configurable
        )
        return val_loader

    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp, bilinear):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # Interpolate resolution based on epoch
        if bilinear:
            # Use bilinear interpolation
            alpha = (epoch - start_ramp) / (end_ramp - start_ramp)
            interp = min_res + alpha * (max_res - min_res)
        else:
            # Use linear interpolation
            interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])[0]

        # Round to nearest multiple of 32
        final_res = int(np.round(interp / 32)) * 32
        return final_res


    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            resolution, distributed, in_memory):
        train_path = Path(train_dataset)
        assert train_path.is_file()

        device = torch.device("cuda", self.trainer.local_rank)

        res_tuple = (resolution, resolution)
        decoder = RandomResizedCropRGBImageDecoder(res_tuple) # Align to torchvision
        image_pipeline = [
            decoder,
            RandomHorizontalFlip(), # Align to torchvision
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True),
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader, decoder

    def create_val_loader(self, val_dataset, num_workers, batch_size,
                            resolution, distributed):
        val_path = Path(val_dataset)
        assert val_path.is_file()

        device = torch.device("cuda", self.trainer.local_rank)

        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True),
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
