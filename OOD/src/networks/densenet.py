"""DenseNet implementation (https://arxiv.org/abs/1608.06993)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
	def __init__(self, n_channels, growth_rate):
		super(Bottleneck, self).__init__()
		inter_channels = 4 * growth_rate
		self.bn1 = nn.BatchNorm2d(n_channels)
		self.conv1 = nn.Conv2d( n_channels, inter_channels, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(inter_channels)
		self.conv2 = nn.Conv2d( inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat((x, out), 1)
		return out

class SingleLayer(nn.Module):
	def __init__(self, n_channels, growth_rate):
		super(SingleLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(n_channels)
		self.conv1 = nn.Conv2d( n_channels, growth_rate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = torch.cat((x, out), 1)
		return out

class Transition(nn.Module):
	def __init__(self, n_channels, n_out_channels):
		super(Transition, self).__init__()
		self.bn1 = nn.BatchNorm2d(n_channels)
		self.conv1 = nn.Conv2d( n_channels, n_out_channels, kernel_size=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = F.avg_pool2d(out, 2)
		return out

class DenseNet(nn.Module):
	def __init__(self, res, growth_rate, depth, reduction, n_classes, bottleneck):
		super().__init__()

		if res[-1] == 32:
			self.factor = 1
		elif res[-1] == 64:
			self.factor = 2

		if bottleneck:
			n_dense_blocks = int((depth - 4) / 6)
		else:
			n_dense_blocks = int((depth - 4) / 3)

		n_channels = 2 * growth_rate
		self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False)

		self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
		n_channels += n_dense_blocks * growth_rate
		n_out_channels = int(math.floor(n_channels * reduction))
		self.trans1 = Transition(n_channels, n_out_channels)

		n_channels = n_out_channels
		self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
		n_channels += n_dense_blocks * growth_rate
		n_out_channels = int(math.floor(n_channels * reduction))
		self.trans2 = Transition(n_channels, n_out_channels)

		n_channels = n_out_channels
		self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
		n_channels += n_dense_blocks * growth_rate

		self.bn1 = nn.BatchNorm2d(n_channels)
		self.fc = nn.Linear(n_channels, n_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def _make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck):
		layers = []
		for _ in range(int(n_dense_blocks)):
			if bottleneck:
				layers.append(Bottleneck(n_channels, growth_rate))
			else:
				layers.append(SingleLayer(n_channels, growth_rate))
			n_channels += growth_rate
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.trans1(self.dense1(out))
		out = self.trans2(self.dense2(out))
		out = self.dense3(out)
		out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), self.factor * 8))
		out = self.fc(out)
		return out

def DenseNet12x40(res, num_classes): return DenseNet(res, growth_rate=12, depth=40, reduction=1.0, n_classes=num_classes, bottleneck=False)
def DenseNet12x100(res, num_classes): return DenseNet(res, growth_rate=12, depth=100, reduction=1.0, n_classes=num_classes, bottleneck=False)
