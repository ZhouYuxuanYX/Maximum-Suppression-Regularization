import torch
import argparse
import lightning as L
import os
import warnings
import logging
from omegaconf import OmegaConf

from src.networks.googlenet import *
from src.networks.resnet_tv import *
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
			'nasnet': NasNet,
			'tvresnet18': tvresnet18,
			'tvresnet34': tvresnet34,
			'tvresnet50': tvresnet50,
			}

AUGMENTATIONS = ('','f','c','r','fcr','z','augmix','mixup','cutmix','trivaug','randaug','autoaug')

# parser
parser = argparse.ArgumentParser(description='Mollified classification')
parser.add_argument('-d', '--data', choices=['cifar10','cifar100','tin'], default='cifar10')
parser.add_argument('-n', '--net', choices=networks.keys(), default='presnet50')
parser.add_argument('-b', '--mb', type=int, default=128)
parser.add_argument('-e', '--epoch', type=int, default=200)
parser.add_argument('-r', '--lr', type=float, default=0.01)
parser.add_argument('-l', '--loss', choices=['ce','nce','bce','maxsup','lsce'], default='ce', help='Cross-entropy, normalized CE, binary CE, MaxSup, or Label-Smoothing CE')
parser.add_argument('--scaling', choices=['normal','uniform'], default='normal', help='Data normalization')
parser.add_argument('-a', '--aug', choices=AUGMENTATIONS, nargs="+", default='', help='Augmentations')
parser.add_argument('-m', '--mollify', action='store_true', default=False, help='Noise inputs and outputs')
parser.add_argument('--ynoise', choices=['ls','lm','la','ld'], default='ls', help='Label smoothing (ls), Label mixing (lb), Label addition (la), or Label degradation (ld)')
parser.add_argument('--npow', type=float, default=1, help="Diffusion label power")
parser.add_argument('--bpow', type=float, default=1, help="Blurring label power")
parser.add_argument('--alpha', type=float, default=1, help="Alpha α parameter")
parser.add_argument('--beta', type=float, default=2, help="Beta β parameter")
parser.add_argument('--batch', type=str, default='xnb', help='Minibatch roles, combination of [x|n|b]')
parser.add_argument('--rep', type=int, default=1, help='Repeated batching')
parser.add_argument('-w', '--wandb', action='store_true', default=False)
parser.add_argument('-c', '--cluster', action='store_true', default=False, help='Cluster mode (suppress output)')
parser.add_argument('--cpus', type=int, default=1)
parser.add_argument('--valfreq', type=int, default=1000, help='Validation frequency over epochs')
args = parser.parse_args()

conf = OmegaConf.load('config')

def args2str(args):
	s = '%s-%s' % (args.data, args.net)
	s += "-L(%s,%d)" % (args.loss,args.epoch)
	if args.aug:
		s += '-A(%s)' % ','.join(args.aug)
	if args.mollify:
		s += '-M(%s,%s,n^%.1f,b^%.1f,r%d)' % (args.batch, args.ynoise, args.npow, args.bpow, args.rep)
		s += '-B(a=%.1f,b=%d)' % (args.alpha, args.beta)
	return s

# main
if __name__=='__main__':
	# load data
	loaders_tr,loaders_val,res,num_classes = load_data( args.data, args.mb, args.scaling, args.aug, args.cpus )

	# create network and Lightning model
	sink = ("la" in args.ynoise) or ("lm" in args.ynoise)
	net = networks[args.net](res, num_classes + sink)
	model = Model(net, args, res, num_classes=num_classes)

	# wandb logging
	wandb_logger = L.pytorch.loggers.WandbLogger( log_model=args.wandb, entity=conf['wandb_entity'], project=conf['wandb_project'], offline=not args.wandb, config=vars(args), name=args2str(args) )

	# lightning trainer parameters
	trainer_args = {
		'max_epochs': args.epoch,
		'log_every_n_steps': 30,
		'check_val_every_n_epoch': args.valfreq,
		'num_sanity_val_steps': 0,
		'logger': wandb_logger,
		'callbacks': [L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')],
		'enable_progress_bar': not args.cluster,
 		'precision': 'bf16-mixed' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '16-mixed',
		'benchmark': True,
		'enable_checkpointing': False
		}

	print( 'args:', vars(args) )
	print( 'name:', args2str(args) )
	print( 'pars:', '%.1fM' % (sum(p.numel() for p in net.parameters() if p.requires_grad)/1000000) )

	# network info
	if torch.cuda.is_available():
		print( 'cuda', '%dx %s' % (torch.cuda.device_count(), torch.cuda.get_device_name(0)) )

	# train
	trainer = L.Trainer( **trainer_args )
	trainer.fit( model, loaders_tr, loaders_val )
	trainer.validate(model, loaders_val)
	trainer.save_checkpoint('checkpoints/%s.ckpt' % args2str(args) )
