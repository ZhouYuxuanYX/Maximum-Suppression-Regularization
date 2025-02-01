import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import lightning as L
import numpy as np
import torchmetrics

from losses import OnlineLabelSmoothing, LogitPenalty, MaxSuppression
import torchvision
from typing import Dict, Any

class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        # Ensure the filter is contiguous and properly strided
        self.register_buffer('blur_filter', filt.contiguous())

    def forward(self, x):
        # Ensure input is contiguous before conv2d operation
        x = x.contiguous()
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                          groups=self.conv.in_channels)
        # Ensure blurred output is contiguous before next conv
        blurred = blurred.contiguous()
        return self.conv(blurred)

class Net(L.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # init the model 
        self.config = config
        # support all model in torchvision models
        self.model = torchvision.models.get_model(
                self.config['model'],
                weights=self.config['weights'] if self.config['weights'] != "" else None,
                num_classes=self.config['num_classes']
            )
        self._apply_blurpool(self.model)
        self.criterion = self._get_criterion()
        self.val_criterion = nn.CrossEntropyLoss()

        #metrics 
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.config['num_classes'], top_k=1)
        self.train_topk_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.config['num_classes'], top_k=5)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.config['num_classes'], top_k=1)
        self.val_topk_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.config['num_classes'], top_k=5)

        self.validation_step_outputs = []
        self.train_step_outputs = []

    def _apply_blurpool(self, mod: torch.nn.Module):
        for (name, child) in mod.named_children():
            if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
                # Ensure the conv layer's parameters are contiguous
                if hasattr(child, 'weight'):
                    child.weight.data = child.weight.data.contiguous()
                if hasattr(child, 'bias') and child.bias is not None:
                    child.bias.data = child.bias.data.contiguous()
                # Create and set BlurPoolConv2d
                blurpool = BlurPoolConv2d(child)
                setattr(mod, name, blurpool)
            else:
                self._apply_blurpool(child)

    def _get_criterion(self):
        if self.config['loss']['loss_type'] == "ce":
            return nn.CrossEntropyLoss()
        elif self.config['loss']['loss_type'] == "ols":
            return OnlineLabelSmoothing(self.config['loss']['alpha'], self.config['loss']['smoothing'])
        elif self.config['loss']['loss_type'] == "ls":
            return nn.CrossEntropyLoss(label_smoothing=self.config['loss']['label_smoothing'])
        elif self.config['loss']['loss_type'] == "lp":
            return LogitPenalty(self.config['loss']['weight'], self.config['loss']['beta'])
        elif self.config['loss']['loss_type'] == "ms":
            return MaxSuppression(self.config['loss']['begin_lambda'], self.config['loss']['end_lambda'], self.config['loss']['epochs'])
        else:
            raise ValueError(f"Criterion {self.config['loss']['loss_type']} not supported")
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        out = self.model(x)
        if self.config['loss']['loss_type'] == "ms":
            self.criterion.set_current_epoch(self.current_epoch)
        loss = self.criterion(out, y)
        self.train_accuracy(out, y)
        self.train_topk_accuracy(out, y)
        self.log('train_loss_step', loss, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_acc_step', self.train_accuracy, prog_bar=False, logger=True)
        self.log('train_topk_acc_step', self.train_topk_accuracy, prog_bar=False, logger=True)
        self.train_step_outputs.append(loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        # While validating, we fix the cross entropy loss
        x, y = batch 
        out = self.model(x)
        loss = self.val_criterion(out, y)
        self.val_accuracy(out, y)
        self.val_topk_accuracy(out, y)
        self.log('val_loss_step', loss, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_acc_step', self.val_accuracy, prog_bar=False, logger=True)
        self.log('val_topk_acc_step', self.val_topk_accuracy, prog_bar=False, logger=True)
        self.validation_step_outputs.append(loss)
        return loss 

    def on_training_epoch_end(self):
        epoch_loss = torch.stack(self.train_step_outputs).mean()
        self.log('Train_Loss', epoch_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('Train_Acc_Top1', self.train_accuracy, prog_bar=True, logger=True)
        self.log('Train_Acc_Top5', self.train_topk_accuracy, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('Val_Loss', epoch_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('Val_Acc_Top1', self.val_accuracy, prog_bar=True, logger=True)
        self.log('Val_Acc_Top5', self.val_topk_accuracy, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        
        # Initialize parameter groups
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Ensure contiguous memory layout for parameters
            param.data = param.data.contiguous()
            
            if 'bn' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.SGD([
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params, 'weight_decay': self.config['optimizer']['weight_decay']}
        ], lr=self.config['optimizer']['lr'], momentum=self.config['optimizer']['momentum'])

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['scheduler']['step_size'],
            gamma=self.config['scheduler']['gamma']
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

