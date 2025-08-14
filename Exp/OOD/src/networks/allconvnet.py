"""AllConv implementation (https://arxiv.org/abs/1412.6806)"""
import math
import torch
import torch.nn as nn

class GELU(nn.Module):
	def forward(self, x):
		return torch.sigmoid(1.702 * x) * x

def make_layers(cfg):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'Md':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5)]
		elif v == 'A':
			layers += [nn.AvgPool2d(kernel_size=8)]
		elif v == 'NIN':
			conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1)
			layers += [conv2d, nn.BatchNorm2d(in_channels), GELU()]
		elif v == 'nopad':
			conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
			layers += [conv2d, nn.BatchNorm2d(in_channels), GELU()]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			layers += [conv2d, nn.BatchNorm2d(v), GELU()]
			in_channels = v
	return nn.Sequential(*layers)

class AllConvNet(nn.Module):
	def __init__(self, res, num_classes):
		super().__init__()
		self.num_classes = num_classes
		self.width1,w1 = 96,96
		self.width2,w2 = 192,192
		self.features = make_layers([w1, w1, w1, 'Md', w2, w2, w2, 'Md', 'nopad', 'NIN', 'NIN', 'A'])
		if res[-1] == 32:
			self.out = nn.Linear(self.width2, num_classes)
		elif res[-1] == 64:
			self.out = nn.Linear(self.width2*4, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.out(x)
		return x
