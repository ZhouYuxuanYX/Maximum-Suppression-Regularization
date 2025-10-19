import torch
import torchvision
import lightning as L
import torch.nn.functional as F
import torchmetrics
import torch_dct
import math

from src.data import *

# main lightning training model
class Model(L.LightningModule):
	def __init__(self, net, args, res, num_classes ):
		super().__init__()
		self.net = net
		self.args = args
		self.res = res
		self.num_classes = num_classes
		self.corruptions = corruptions
		self.scalemin = 0.3
		self.scalemax = res[-1]

		freqs = torch.pi * torch.arange(0,res[-1]).reshape((res[-1],1)) / res[-1]
		self.L = freqs**2 + freqs.T**2

		# torch_dct contains nn.Module's that compute the dct without fouriers
		self.dctlayer  = torch_dct.LinearDCT( res[-1], 'dct' )
		self.idctlayer = torch_dct.LinearDCT( res[-1], 'idct' )

		self.save_hyperparameters()

	def forward(self, x):
		return self.net(x)

	# redefine logger to support None's
	def log(self, *args, **kwargs):
		if args[1] is not None:
			super().log(*args,**kwargs)

	# SGD + Cosine
	def configure_optimizers(self):
		opt = torch.optim.SGD( self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
		sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.epoch)
		return [opt],[sch]

	def onehot(self, y):
		sink = ("la" in self.args.ynoise) or ("lm" in self.args.ynoise)
		y = F.one_hot(y, num_classes=self.num_classes + sink).float()
		return y
	
	# minibatch statistics
	def summary(self, x, y):
		logits = self.forward(x) # (N,P)
		probs = logits.softmax(1) # (N,P)
		fmax = logits[:,0:self.num_classes].argmax(1)
		ymax = y[:,0:self.num_classes].argmax(1)

		error = 1 - (fmax == ymax).sum() / y.size(0)

		ece = None
		if torch.backends.mps.is_available() == False:
			ece = torchmetrics.functional.calibration_error(probs, ymax, 'multiclass', num_classes=self.num_classes)
		
		# noise MAE if exists
		noisemae = None
		if y.shape[1] % 2 == 1: # odd
			noisemae = (probs[:,-1] - y[:,-1]).abs().mean()

		if self.args.loss == 'ce':
			nll = F.cross_entropy( logits, y )
		elif self.args.loss == 'nce':
			nll = self.normalized_cross_entropy( logits, y )
		elif self.args.loss == 'bce':
			nll = F.binary_cross_entropy_with_logits(logits, y)
		elif self.args.loss == 'maxsup':
			loss_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
			z_top1 = logits.topk(1, -1)[0]
			reg = z_top1 - logits.mean(-1, keepdim=True)
			target = y.argmax(dim=-1)  # convert one-hot to class indices
			lam = 0.1 + 0.1 * self.current_epoch / max(self.args.epoch - 1, 1)
			nll = loss_criterion(logits, target) + lam * reg.mean()
		elif self.args.loss == 'lsce':
			loss_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
			target = y.argmax(dim=-1)
			nll = loss_criterion(logits, target)
		
		return nll,error,ece,noisemae

	def normalized_cross_entropy(self, logits, y):
		# stable way of computing geometric mean
		f = logits.softmax(1)
		logf = logits.log_softmax(1) # (N,K)
		logfp = logf.mean(1,True)    # (N,1)
		fp = logfp.exp()             # (N,1)
		Z = ( (fp-f) / (logfp - logf) ).sum(1).log()
		nll = -(y*logf).sum(1).mean() + Z.mean()
		return nll

	# sample batch of temperatures from Beta(a,b)
	def sample_time(self, x, roles):
		B = torch.distributions.beta.Beta(self.args.alpha, self.args.beta)
		t = B.sample( [x.shape[0]] ).type_as(x)

		t[roles == 'x'] = 0   # clip clean and pure noise to 0 and 1
		t[roles == 'e'] = 1

		return t

	def dct(self, x):
		return torch_dct.apply_linear_2d(x, self.dctlayer)
	
	def idct(self, x):
		return torch_dct.apply_linear_2d(x, self.idctlayer)

	# logscale from Rissanen et al (2023)
	def blurscale(self, t):
		return torch.exp( (1-t)*math.log(self.scalemin) + t * math.log(self.scalemax))

	def dissipation_time(self, t):
		return self.blurscale(t)**2 / 2

	def xblur(self, x, t):
		A = torch.exp( -self.L.type_as(x)[None,None,:,:] * self.dissipation_time(t)[:,None,None,None] )
		u = self.dct(x)
		xt = self.idct( A * u )
		return xt

	def alpha(self,t):
		return torch.cos( 0.5 * torch.pi * t )

	def sigma(self,t):
		return torch.sin( 0.5 * torch.pi * t )

	def snr(self, t):
		return self.alpha(t)**2 / self.sigma(t)**2

	def gamma_noise(self, t):
		return 1 / (1 + self.snr(t))**self.args.npow

	def gamma_blur(self, t):
		return t**self.args.bpow

	def xnoise(self, x, t):
		eps = torch.randn_like(x)
		xt = self.alpha(t)[:,None,None,None] * x + self.sigma(t)[:,None,None,None] * eps
		return xt
	
	def sample_role(self, x):
		return np.random.choice( list(self.args.batch), x.shape[0] )

	# noise inputs
	def mollify_x(self, x, t, roles):
		if 'n' in roles:
			x[roles=='n'] = self.xnoise( x[roles=='n'], t[roles=='n'] )
		if 'b' in roles:
			x[roles=='b'] = self.xblur( x[roles=='b'], t[roles=='b'] ).type_as(x)  # AMP casts blur to half, but does not do it for x
		if 'e' in roles:
			x[roles=='e'] = torch.randn_like( x[roles=='e'] )
		return x

	def label_decay(self, y, t):
		# label smoothing
		if self.args.ynoise == 'ls':
			u = torch.ones_like(y) / y.shape[1]
			y = t[:,None] * u + (1-t)[:,None] * y

		# label degradation
		elif self.args.ynoise == 'ld':
			y = (1-t)[:,None] * y

		# add new label
		elif self.args.ynoise == 'la':
			y[:,self.num_classes:(self.num_classes+1)] = t

		# add label + smooth
		elif self.args.ynoise == 'lm':
			y[:,self.num_classes:(self.num_classes+1)] = t
			y[:,0:self.num_classes] *= 1-t

		return y

	# mix noise into labels
	def mollify_y(self, y, t, roles):
		if 'n' in roles:
			y[roles=='n'] = self.label_decay( y[roles=='n'], self.gamma_noise( t[roles=='n'] ) )
		if 'b' in roles:
			y[roles=='b'] = self.label_decay( y[roles=='b'], self.gamma_blur( t[roles=='b'] ) ) # assume linear ls
		if 'e' in roles:
			y[roles=='e'] = torch.ones_like(y[roles=='e']) / y.shape[1]

		return y

	def training_step(self, batch, batch_idx):
		x,y = batch

		if 'mixup' in self.args.aug:
			mixup = torchvision.transforms.v2.MixUp(num_classes=self.num_classes)
			x,y = mixup(x,y)
		elif 'cutmix' in self.args.aug:
			cutmix = torchvision.transforms.v2.CutMix(num_classes=self.num_classes)
			x,y = cutmix(x,y)
		else:
			y = self.onehot(y)
  
		# repeated sampling
		if self.args.rep > 1:
			repidx = torch.arange(self.args.mb/self.args.rep, dtype=torch.long).repeat(self.args.rep)
			x = x[repidx]
			y = y[repidx]

		if self.args.mollify:
			roles = self.sample_role(x)
			t = self.sample_time(x, roles )
			x = self.mollify_x(x, t, roles )
			y = self.mollify_y(y, t, roles )

#			self.logger.experiment.log( {"temperature" : t} ) # workaround to log histograms

		nll,err,ece,noisemae = self.summary(x,y)
		
		self.log('nll/train', nll, prog_bar=True )
		self.log('error/train', err )
		self.log('ece/train', ece )
		self.log('noise_error/train', noisemae )
		self.log('beta', self.args.beta )

		return nll

	def validation_step(self, batch, batch_idx, dataloader_idx=0):
		x,y = batch
		y = self.onehot(y)

		nll,err,ece,noisemae = self.summary(x, y)  # dataloader_idx tells the corruption

		# clean
		if dataloader_idx == 0:
			self.log('nll', nll, add_dataloader_idx=False, sync_dist=True)
			self.log('error', err, add_dataloader_idx=False, sync_dist=True)
			self.log('ece', ece, add_dataloader_idx=False, sync_dist=True )
			self.log('noise_error', noisemae, add_dataloader_idx=False, sync_dist=True )
		
		# corrs 
		if dataloader_idx > 0:
			intensity = 1 + batch_idx // 100   # assumes batch_size 100
			self.log('nll/%s_%d' % (self.corruptions[dataloader_idx-1],intensity), nll, add_dataloader_idx=False, sync_dist=True )
			self.log('error/%s_%d' % (self.corruptions[dataloader_idx-1],intensity), err, add_dataloader_idx=False, sync_dist=True )
			self.log('ece/%s_%d' % (self.corruptions[dataloader_idx-1],intensity), ece, add_dataloader_idx=False, sync_dist=True )
