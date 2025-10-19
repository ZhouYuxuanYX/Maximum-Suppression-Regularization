# ./loss/MaxSupLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxSupLoss(nn.Module):
    def __init__(self, num_epochs=200):
        super().__init__()
        self.epoch = 0
        self.total_epochs = num_epochs
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.0)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, target):
        loss_ce = self.ce(x, target)
        
        z_top1 = x.topk(1, dim=-1)[0]
        reg = z_top1 - x.mean(dim=-1, keepdim=True)
        lam = 0.1 + 0.1 * self.epoch / max(1, self.total_epochs - 1)
        return loss_ce + lam * reg.mean()

def create_loss(num_epochs=200):
    print(f'Loading MaxSup Loss with total_epochs={num_epochs}')
    return MaxSupLoss(num_epochs=num_epochs)