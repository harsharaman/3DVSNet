import argparse
import tqdm
import os
import yaml
import time
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models import *

from utils.metrics import runningScore
from utils.losses import crossentropy, focalloss
from data_loader.dataloader import *

def parse_args():
	parser = argparse.ArgumentParser()

	#model
	parser.add_argument('--name', default='baseline',
				help="model name: SR_VSNet, SR_VSNet_scSE, SR_VSNet_separable, SR_VSNet_separable_scSE",
				choices=['SR_VSNet', 'SR_VSNet_scSE', 'SR_VSNet_separable_scSE', 'SR_VSNet_seperable'])
	parser.add_argument('--epochs', default=100, type=int, metavar='N',
				help="number of epochs to run")
	parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N',
				help="mini-batch size (default: 2)")
	parser.add_argument('--early_stopping', default=40, type=int, metavar='N',
				help="early-stopping (default: 40)")
	parser.add_argument('--num_workers', default=8, type=int)
	parser.add_argument('--loss',default='ce', choices=['ce','fl','reg_fl'], type=str
				help="which loss function to use. Choice between Categorical Cross Entropy, Focal Loss and Regularized Focal Loss")
	parser.add_argument('--alpha', default=[1,1,1], type=list,
				help = "alpha modulation for focal loss")
	parser.add_argument('-rw','--reg_wt', default=1e-3, type=float,
				help='weight for regularization term for focal loss')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam','SGD'],
				help="Optimizer (default: Adam)")
	parser.add_argument('--lr', default=1e-2, type=float, metavar='LR',
				help="initial learning rate")
	parser.add_argument('--weight_decay', default=1e-4, type=float, 
				help="weight decay")
	
	config = parser.parse_args()

	return config

def train(trainloader, net, criterion, optimizer):
	training_start_time = time.time()
	net.train()
	
	#initialization
	n_classes = 3
	best_iou = -100.0
	best_dice = 0
	train_losses, val_losses= [], []
	running_loss = 0

	pbar = tqdm(total=len(trainloader), desc='Training')

	for tr_idx, data_samples in (enumerate(trainloader)):
        	optimizer.zero_grad()
        	volume, labels = data_samples['data'], data_samples['target']
        	volume = volume.cuda()
        	labels = labels.long().cuda()
        	outputs = net(volume)
        	loss = criterion(input_=outputs, target =labels) 

        	loss.backward()
        	optimizer.step()

        	running_loss += loss.item()
		pbar.update(1)
	pbar.close()
	
	return running_loss / (2 * tr_idx)

def validate(valloader, net, criterion):
	#validate
	net.eval()

	#initialization
	n_classes = 3
	val_loss = 0
	running_metrics = runningScore(n_classes)
	running_metrics.reset()

	with torch.no_grad():
		pbar = tqdm(total=len(valloader),desc='Validation')
		
		for val_idx, data_samples in enumerate(valloader):
			volume, labels = valloader['data'], valloader['target']
			volume = volume.cuda()

			labels = labels.long().cuda()
                
                	outputs = net(volume)
                	validation_loss_current_model = criterion(input_=outputs, target =labels)
                	val_loss += criterion(input_=outputs, target =labels)
                	pred = outputs.data.max(1)[1].cpu().numpy()
                	gt = labels.cpu().numpy()
                	running_metrics.update(gt, pred)

			pbar.update(1)
		pbar.close()

	return val_loss/(val_idx*2), running_metrics.get_scores()

def main():
	#Get configuration
	config = vars(parse_args())

	#Make model output directory
	file_name = config['name']

	os.makedirs('model_outputs/{}'.format(file_name),exist_ok=True)

	print("Creating directory called {}".format(file_name))

	print('-' * 20)
	print("Configuration settings as follows:")
	for key in config:
		print('{}: {}'.format(key, config[key]))
	print('-' * 20)

	#Save configuration
	with open('model_outputs/{}/config.yml'.format(file_name), 'w') as f:
		yaml.dump(config, f)

	#Criterion
	if config['loss'] == 'ce':
		criterion = categorical_cross_entropy()
	elif config['loss'] == 'fl':
		criterion = FocalLoss(alphas=config['alpha'], gamma=2.0)
	elif config['loss'] == 'reg_fl':
		criterion = RegulatedFocalLoss(alphas=[1.,1.,1.], gamma=2.0, weight=1e-4, modified=False) # Must add Gaussian
	else: print('Select a valid loss')

	cudnn.benchmark = True

	#Create model
	print("=> Creating model")
	if config['name'] == 'SR_VSNet':
		model = SR_VSNet()
	

