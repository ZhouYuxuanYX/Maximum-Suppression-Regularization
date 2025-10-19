# ResNet
#
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385v1

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()

        self.resfunc = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out * ResidualBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out * ResidualBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != ResidualBlock.expansion * c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out * ResidualBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out * ResidualBlock.expansion)
            )

    def forward(self, x):
        return F.relu(self.resfunc(x) + self.shortcut(x), inplace=True)

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.resfunc = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or c_in != c_out * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out * BottleNeck.expansion)
            )

    def forward(self, x):
        return F.relu(self.resfunc(x) + self.shortcut(x), inplace=True)

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.c_base = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.stage1 = self._make_layer(block, 64, num_block[0], 1)
        self.stage2 = self._make_layer(block, 128, num_block[1], 2)
        self.stage3 = self._make_layer(block, 256, num_block[2], 2)
        self.stage4 = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.c_base, out_channels, stride))
            self.c_base = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(res, num_classes): return ResNet(ResidualBlock, [2,2,2,2], num_classes)
def ResNet34(res, num_classes): return ResNet(ResidualBlock, [3,4,6,3], num_classes)
def ResNet50(res, num_classes): return ResNet(BottleNeck, [3,4,6,3], num_classes)
def ResNet101(res, num_classes): return ResNet(BottleNeck, [3,4,23,3], num_classes)
def ResNet152(res, num_classes): return ResNet(BottleNeck, [3,8,36,3], num_classes)
