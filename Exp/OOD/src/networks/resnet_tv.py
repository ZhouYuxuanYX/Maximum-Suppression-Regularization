import torch.nn as nn
import torchvision

TVRESNETS = {
    18: torchvision.models.resnet18,
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50
    }

# change last layer of Torchvision resnets
class ResNetTV(nn.Module):
    def __init__(self, layers, num_classes, pretrain=True):
        super().__init__()

        init = 'DEFAULT' if pretrain else None
        self.net = TVRESNETS[layers](weights=init)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)

def tvresnet18(num_classes, pretrain=True): return ResNetTV(18, num_classes, pretrain)
def tvresnet34(num_classes, pretrain=True): return ResNetTV(34, num_classes, pretrain)
def tvresnet50(num_classes, pretrain=True): return ResNetTV(50, num_classes, pretrain)
