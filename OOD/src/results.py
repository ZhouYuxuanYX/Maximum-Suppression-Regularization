# Script to extract results from optimised checkpoints
# Triton does not support gpu-jupyter, so we have to extract things into csv and plot later

import torch
import torchinfo
import argparse
import lightning as L
import os
import warnings
import logging
import torch.nn.functional as F

import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt

from src.networks.googlenet import *
from src.networks.resnet import *
from src.networks.presnet import *
from src.networks.allconvnet import *
from src.networks.densenet import *
from src.networks.resnext import *
from src.networks.wrn import *
from src.networks.nasnet import *

from src.data import *
from src.model import *

os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore", ".*does not have many workers.*")
torch.set_float32_matmul_precision('high')
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # silent lightning

networks = {'allconvnet10': AllConvNet,
			'densenet40': DenseNet12x40, 
			'densenet100': DenseNet12x100,
			'wrn40': WRN40x2, 
			'wrn40_10': WRN40x10,
			'resnext29': ResNext29,
			'presnet18': PreActResNet18,
			'presnet34': PreActResNet34,
			'presnet50': PreActResNet50,
			'presnet101': PreActResNet101,
			'resnet18': ResNet18,
			'resnet34': ResNet34,
			'resnet50': ResNet50,
			'resnet101': ResNet101,
			'googlenet': GoogleNet,
			'nasnet': NasNet}

corruptions = ('brightness',
			   'contrast',
			   'defocus_blur',
			   'elastic_transform',
			   'fog',
			   'frost',
			   'gaussian_noise',
			   'glass_blur',
			   'impulse_noise',
			   'jpeg_compression',
			   'motion_blur',
			   'pixelate',
			   'shot_noise',
			   'snow',
			   'zoom_blur')

noise   = ('shot_noise','impulse_noise','gaussian_noise')
blur    = ('motion_blur','zoom_blur','defocus_blur','glass_blur')
weather = ('fog','frost','snow','brightness')
digital = ('jpeg_compression','pixelate','elastic_transform','contrast')

def load_model():
	# load data
	loader_tr,loaders_val,res,classes = load_data( 'cifar10', 128, 'normal' )

	# create network and Lightning model
	net = networks['presnet18'](classes)
	model = Model(net, None, num_classes=classes)

	fn = 'ProbAug/2r9wmhlq/checkpoints/epoch=299-step=117300.ckpt'

	checkpoint = torch.load(fn, map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['state_dict'])

	return loader_tr,loaders_val,model

def load_batch(loader, N=128, K=1):
	mb_idx = torch.randint(0, 50000, size=(N,))
	x = torch.zeros((N,K,3,32,32))
	y = torch.zeros((N,K))
	for i in range(N):
		for j in range(K):
			x[i,j,:,:,:],y[i,j] = loader.dataset[mb_idx[i]]
	
	x = x.view((N*K,3,32,32))
	y = y.view((N*K))

	return x,y

def evaluate_likelihood(model, loader, Nrep = 50):
	#Ns = (8,16,32,64,128,256,512,1024)
	#Ks = (1,2,4,8,16,32,64,128)
	
	Ns = (8,16,32,64,128)
	Ks = (1,2,4,8)
	L = torch.zeros( (len(Ns),len(Ks)) )

	# repeat
	for i,N in enumerate(Ns):
		for j,K in enumerate(Ks):
			for r in range(Nrep):
				x,y = load_batch(loader, N,K)
				y = F.one_hot(y.long(), num_classes=10).float()
				logits = model(x)
				ll = -F.cross_entropy( logits, y )
				L[i,j] += ll / (K*Nrep)
	return L


def extract_wandb():
	runs = wandb.Api().runs("markus-heinonen/ProbAug")

	metrics,configs,names = [],[],[]
	for run in runs: 
		metrics.append( run.summary._json_dict )
		configs.append( {k: v for k,v in run.config.items() if not k.startswith('_')} )
		names.append( run.name )
	df = pd.DataFrame( {"log": metrics, "config": configs, "name": names} )
	df.to_csv('results.csv')

	return metrics,configs,names

def print_summary(metrics, configs, names):
	for i in range(len(names)):
		id = names[i]

		try:
			nll = metrics[i]['nll']
			err = metrics[i]['error']
			ece = metrics[i]['ece']
			
			cnll = np.mean([metrics[i]['nll/%s_%d' % (c,k+1)] for c in corruptions for k in range(5) ])
			cerr = np.mean([metrics[i]['error/%s_%d' % (c,k+1)] for c in corruptions for k in range(5) ])
			cece = np.mean([metrics[i]['ece/%s_%d' % (c,k+1)] for c in corruptions for k in range(5) ])

			print( '%03d | nll %.3f err %.3f ece %.3f cnll %.3f cerr %.3f cece %.3f | %s %d' % (i+1, nll,err,ece,cnll,cerr,cece,id, configs[i]['beta']) )

		except:
			pass

	#nll = [metrics[i]['nll'] for i in range(24) ]
	#err = [metrics[i]['error'] for i in range(24) ]
	#ece = [metrics[i]['ece'] for i in range(24) ]

	#cnll = [ np.mean([metrics[i]['nll/%s_%d' % (c,k)] for c in corruptions_15 for k in range(5) ]) for i in range(24) ]
	#cerr = [ np.mean([metrics[i]['error/%s_%d' % (c,k)] for c in corruptions_15 for k in range(5) ]) for i in range(24) ]
	#cece = [ np.mean([metrics[i]['ece/%s_%d' % (c,k)] for c in corruptions_15 for k in range(5) ]) for i in range(24) ]


# x-axis: moll magnitude (beta)
# y-axis: (nll/err/ece) vs (cnll,cerr,cece)
# lines: fcr+m model 
def plot_betas():
	pass

# x-axis: corruption level 0,1,2,3,4,5
# y-axis: nll/err/ece
# lines: fcr/fcr+m vs fcr+augmix/fcr+augmix+m vs ''/m
def plot_corrlevels():
	pass

# 
def plot_():
	pass


metrics,configs,names = extract_wandb()
print_summary(metrics, configs, names)

#loader_tr,loaders_val,model = load_model()
#L = evaluate_likelihood(model, loader_tr, Nrep = 50)

