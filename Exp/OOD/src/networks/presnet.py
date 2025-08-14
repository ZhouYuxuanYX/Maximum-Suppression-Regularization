# ResNet
#
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385v1

import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
	expansion = 1

	def __init__(self, c_in, c_out, stride=1):
		super().__init__()

		self.layer = nn.Sequential(
			nn.BatchNorm2d(c_in),
			nn.ReLU(),
			nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(c_out),
			nn.ReLU(),
			nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
		)

		self.shortcut = nn.Sequential()
		if stride != 1 or c_in != self.expansion*c_out:
			self.shortcut = nn.Conv2d(c_in, self.expansion*c_out, kernel_size=1, stride=stride, bias=False)

	def forward(self, x):
		return self.layer(x) + self.shortcut(x)

class PreActBottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super().__init__()

		self.resfunc = nn.Sequential(
			nn.BatchNorm2d(in_planes),
			nn.ReLU(),
			nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
			nn.BatchNorm2d(planes),
			nn.ReLU(),
			nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(planes),
			nn.ReLU(),
			nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		)
		self.shortcut = nn.Sequential()

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

	def forward(self, x):
		return self.resfunc(x) + self.shortcut(x)


class PreActResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super().__init__()

		self.c_base = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # 64x64
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 64x64
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 32x32
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 16x16
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 8x8
		self.avgpool = nn.AdaptiveAvgPool2d( (1,1) )
		self.linear = nn.Linear(512*block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.c_base, planes, stride))
			self.c_base = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avgpool(out).flatten(1)
		out = self.linear(out)
		return out

def PreActResNet18(res, num_classes): return PreActResNet(PreActBlock, [2,2,2,2], num_classes)
def PreActResNet34(res, num_classes): return PreActResNet(PreActBlock, [3,4,6,3], num_classes)
def PreActResNet50(res, num_classes): return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes)
def PreActResNet50(res, num_classes): return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes)
def PreActResNet101(res, num_classes): return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes)
