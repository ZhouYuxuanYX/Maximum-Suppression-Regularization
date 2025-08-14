"""
Assumes files have already been downloaded and placed into data/ folder

- CIFAR-10        https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR-100       https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
- TinyImageNet    http://cs231n.stanford.edu/tiny-imagenet-200.zip
- CIFAR-10-C      https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
- CIFAR-100-C     https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
- TinyImageNet-C  https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar

This script processes these datasets into folders

- data/CIFAR-10
- data/CIFAR-100
- data/TIN

with .npy files

 - train.npy (N,32,32,3) uint8
 - val.npy (10000,32,32,3) uint8
 - {corr}.npy (10000,32,32,3) uint8 
       for 15 corruptions, each having 10000 images with severities 1,..,5 in this order
 - train_labels.npy (N) uint8
 - val_labels.npy (10000) uint8 
 - corr_labels.npy (10000) uint8

where CIFAR has N=50,000 and TIN has N=100,000.
"""

from PIL import Image
import numpy as np
import os
import pickle
import argparse

CIFAR10 = "cifar-10-python.tar.gz"
CIFAR100 = "cifar-100-python.tar.gz"
TIN = "tiny-imagenet-200.zip"

CIFAR10C = "CIFAR-10-C.tar"
CIFAR100C = "CIFAR-100-C.tar"
TINC = "Tiny-ImageNet-C.tar"

CORRS = ('brightness','contrast','defocus_blur','elastic_transform','fog',
		'frost','gaussian_noise','glass_blur','impulse_noise','jpeg_compression',
		'motion_blur','pixelate','shot_noise','snow','zoom_blur')

# parser
parser = argparse.ArgumentParser(description='Process datasets into uint8 .npy files')
parser.add_argument('datasets', type=str, nargs='+' )
parser.add_argument('--folder', type=str, default='.' )
args = parser.parse_args()

datafolder = args.folder


def process_cifar10():
	if not os.path.exists(datafolder + '/CIFAR-10'):
		os.mkdir(datafolder + '/CIFAR-10')

	os.system('tar -xzkf cifar-10-python.tar.gz')
	os.system('tar -xkf CIFAR-10-C.tar')
		
	N = 50000
	x = np.zeros( (N,3,32,32), np.uint8)
	y = np.zeros(  N, np.uint8 )
	for i in range(1,6):
		fn = 'cifar-10-batches-py/data_batch_%d' % i
		I = np.arange( (i-1)*10000, i*10000 )
		with open(fn, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			x[I,:,:,:] = dict[b'data'].reshape((-1,3,32,32))
			y[I] = dict[b'labels']
	
	x = x.transpose((0,2,3,1))
	np.save(datafolder + '/CIFAR-10/train.npy', x)
	np.save(datafolder + '/CIFAR-10/train_labels.npy', y)

	N = 10000
	x = np.zeros( (N,3,32,32), np.uint8)
	y = np.zeros(  N, np.uint8 )
	fn = 'cifar-10-batches-py/test_batch'
	with open(fn, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		x = dict[b'data'].reshape( (-1,3,32,32) )
		y = dict[b'labels']
	
	x = x.transpose((0,2,3,1))
	np.save(datafolder + '/CIFAR-10/val.npy', x)
	np.save(datafolder + '/CIFAR-10/val_labels.npy', y)

	os.system('mv CIFAR-10-C/labels.npy %s/CIFAR-10/corr_labels.npy' % datafolder)
	os.system('mv CIFAR-10-C/* %s/CIFAR-10/' % datafolder)

	os.system('rm %s/CIFAR-10/speckle_noise.npy' % datafolder) 
	os.system('rm %s/CIFAR-10/spatter.npy' % datafolder) 
	os.system('rm %s/CIFAR-10/saturate.npy' % datafolder) 
	os.system('rm %s/CIFAR-10/gaussian_blur.npy' % datafolder) 

	os.system('rm -r cifar-10-batches-py')
	os.system('rm -r CIFAR-10-C')
	
def process_cifar100():
	if not os.path.exists('../data/CIFAR-100'):
		os.mkdir('../data/CIFAR-100')

	os.system('tar -xzkf cifar-100-python.tar.gz')
	os.system('tar -xkf CIFAR-100-C.tar')
	
	fn = 'cifar-100-python/train'
	with open(fn, 'rb') as fo:
		D = pickle.load(fo, encoding='bytes')

	x = D[b'data'].reshape((-1,3,32,32))
	y = D[b'fine_labels']
	x = x.transpose((0,2,3,1))

	np.save(datafolder + '/CIFAR-100/train.npy', x)
	np.save(datafolder + '/CIFAR-100/train_labels.npy', y)

	fn = 'cifar-100-python/test'
	with open(fn, 'rb') as fo:
		D = pickle.load(fo, encoding='bytes')

	x = D[b'data'].reshape((-1,3,32,32))
	y = D[b'fine_labels']
	x = x.transpose((0,2,3,1))

	np.save(datafolder + '/CIFAR-100/val.npy', x)
	np.save(datafolder + '/CIFAR-100/val_labels.npy', y)

	os.system('mv CIFAR-100-C/labels.npy %s/CIFAR-100/corr_labels.npy' % datafolder)
	os.system('mv CIFAR-100-C/* %s/CIFAR-100/' % datafolder)

	os.system('rm %s/CIFAR-100/speckle_noise.npy' % datafolder) 
	os.system('rm %s/CIFAR-100/spatter.npy' % datafolder) 
	os.system('rm %s/CIFAR-100/saturate.npy' % datafolder) 
	os.system('rm %s/CIFAR-100/gaussian_blur.npy' % datafolder) 

	os.system('rm -r CIFAR-100-C')
	os.system('rm -r cifar-100-python')


def process_tin():
	if not os.path.exists(datafolder + '/TIN'):
		os.mkdir(datafolder + '/TIN')

	os.system('unzip -nq tiny-imagenet-200.zip')
	os.system('tar -xkf Tiny-ImageNet-C.tar')

	os.system('cp tiny-imagenet-200/wnids.txt %s/TIN/' % datafolder)
	os.system('cp tiny-imagenet-200/words.txt %s/TIN/' % datafolder)

	### sorted labels
	nids = sorted(open('%s/TIN/wnids.txt' % datafolder).read().splitlines() )
	nid2num = { nid:nids.index(nid) for nid in nids }

	names = {}
	f = open('%s/TIN/words.txt' % datafolder)
	for l in f:
		words = l.split('\t')
		names[ words[0].strip() ] = words[1].strip()
	
	f.close()

	f = open('%s/TIN/labels.txt' % datafolder, 'w')
	for i,nid in enumerate(nids):
		f.write('%d\t%s\t%s\n' % (i,nid,names[nid]) )
	f.close()

	### validation
	print('val')
	x = np.zeros( (10000,64,64,3), np.uint8)
	y = np.zeros(  10000, np.uint8 )

	f = open('tiny-imagenet-200/val/val_annotations.txt')
	for i in range(10000):
		words = f.readline().strip().split('\t')
		fn = 'tiny-imagenet-200/val/images/' + words[0]
		nid = words[1]

		im = Image.open(fn)
		ar = np.asarray(im)

		if ar.ndim == 2:
			x[i,:,:,:] = ar[:,:,None]
		elif ar.ndim == 3:
			x[i,:,:,:] = ar
		
		y[i] = nid2num[nid]

	f.close()

	np.save(datafolder + '/TIN/val.npy', x)
	np.save(datafolder + '/TIN/val_labels.npy', y)

	### train: 100K train images in 200 label-specific folders 
	print('train')
	x = np.zeros( (100000,64,64,3), np.uint8)
	y = np.zeros(  100000, np.uint8 )
	i = 0
	for nid in nids:
		for j in range(500):
			fn = 'tiny-imagenet-200/train/%s/images/%s_%d.JPEG' % (nid,nid,j)
			with Image.open(fn) as im:
				ar = np.asarray(im)

				if ar.ndim == 2:
					x[i,:,:,:] = ar[:,:,None]
				elif ar.ndim == 3:
					x[i,:,:,:] = ar

			y[i] = nid2num[nid]    # all labels are same
			i += 1

	np.save(datafolder + '/TIN/train.npy', x)
	np.save(datafolder + '/TIN/train_labels.npy', y)

	### corrupted val
	# there are 15 corruptions x 5 intensities for 75 types
	# there are 200 labels with 50 images each for 10K images 

	for c in CORRS:
		print('corr', c)
		if os.path.exists('Tiny-ImageNet-C/%s' % c ):
			x = np.zeros( (50000,64,64,3), np.uint8)
			y = np.zeros(  50000, np.uint8 )
			i = 0
			for k in range(1,6):
				for nid in nids:
					path = 'Tiny-ImageNet-C/%s/%d/%s/' % (c,k,nid)
					for fn in os.listdir( path ):
						im = Image.open( path + fn)
						ar = np.asarray(im)

						if ar.ndim == 2:
							x[i,:,:,:] = ar[:,:,None]
						elif ar.ndim == 3:
							x[i,:,:,:] = ar

						y[i] = nid2num[nid]    # all labels are same
						i += 1
				
			np.save(datafolder + '/TIN/%s.npy' % c, x)
			np.save(datafolder + '/TIN/corr_labels.npy', y)

	### delete image files

	os.system('rm -r tiny-imagenet-200')
	os.system('rm -r Tiny-ImageNet-C')


# parser
parser = argparse.ArgumentParser(description='Process datasets into uint8 .npy files')
parser.add_argument('datasets', type=str, nargs='+' )
parser.add_argument('--folder', type=str, default='../data' )
args = parser.parse_args()

if 'cifar10' in args.datasets:
	process_cifar10()

if 'cifar100' in args.datasets:
	process_cifar100()

if 'tin' in args.datasets:
	process_tin()


