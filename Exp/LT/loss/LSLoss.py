# ./loss/LSLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.cls = classes
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        logprobs = self.log_softmax(x)
        true_dist = torch.zeros_like(logprobs)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=1))

def create_loss(num_classes=10, smoothing=0.1):
    print(f'Loading Label Smoothing Loss: ε={smoothing}')
    return LabelSmoothingLoss(classes=num_classes, smoothing=smoothing)