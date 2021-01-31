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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models import vsnet_parts
from models.vsnet import vsnet
from models.vsnet_with_attention import vsnet_with_attention
from models.vsnet_depthwise import vsnet_depthwise

from utils.metrics import runningScore
from utils.losses.crossentropy import categorical_cross_entropy
from utils.losses.focalloss import *
from data_loader.dataloader import *

def parse_args():
	parser = argparse.ArgumentParser()

	#model
	parser.add_argument('--name', default='vsnet',
				help="model name: VSNet, VSNet_with_attention",
				choices=['vsnet', 'vsnet_with_attention','vsnet_depthwise'])
	parser.add_argument('--epochs', default=1000, type=int, metavar='N',
				help="number of epochs to run")
	parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N',
				help="mini-batch size (default: 2)")
	parser.add_argument('--early_stopping', default=40, type=int, metavar='N',
				help="early-stopping (default: 40)")
	parser.add_argument('--num_workers', default=8, type=int)
	parser.add_argument('--loss',default='ce', choices=['ce','fl','reg_fl'], type=str,
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

	pbar = tqdm(total=250, desc='Training')

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
	training_loss = running_loss / (2 * tr_idx)
	#print("Training Loss: {}".format(training_loss))
	return training_loss

def validate(valloader, net, criterion):
	#validate
	net.eval()

	#initialization
	n_classes = 3
	val_loss = 0
	running_metrics = runningScore(n_classes)
	running_metrics.reset()

	with torch.no_grad():
		pbar = tqdm(total=250,desc='Validation')
		
		for val_idx, data_samples in enumerate(valloader):
			volume, labels = data_samples['data'], data_samples['target']
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
	validation_loss = val_loss / (2 * val_idx)
	#print("Training Loss: {}".format(validation_loss))
	return (validation_loss, running_metrics.get_scores())

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
	if config['name'] == 'vsnet':
		model = vsnet()
	elif config['name'] == 'vsnet_with_attention':
		model = vsnet_with_attention()
	elif config['name'] == 'vsnet_depthwise':
		model = vsnet_depthwise()

	model = model.cuda()

	if torch.cuda.device_count() > 1:
		print("Let's use {} GPUs!".format(torch.cuda.device_count()))
		model = nn.DataParallel(model)

	params = filter(lambda p: p.requires_grad, model.parameters())

	if config['optimizer'] == 'Adam':
		optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
		poly_lr = lambda epoch, max_epochs=config['epochs'], initial_lr=config['lr']: initial_lr * (1 - epoch/max_epochs) ** 0.9
		scheduler = LambdaLR(optimizer, lr_lambda=poly_lr)
	elif config['optimizer'] == 'SGD':
		optimizer = optim.SGD(params, lr=config['lr'], weight_decay=config['weight_decay'])
		poly_lr = lambda epoch, max_epochs=config['epochs'], initial_lr=config['lr']: initial_lr * (1 - epoch/max_epochs) ** 0.9
		scheduler = LambdaLR(optimizer, lr_lambda=poly_lr)
	else:
		raise NotImplementedError

	#Dataset loading
	t = "Task003_Liver"
	p = os.path.join("/net/pasnas01/pool1/ramanha/corpus/nnUNet/nnUNet_preprocessed", t)
	with open(os.path.join(p, "nnUNetPlansv2.1_plans_3D.pkl"), 'rb') as f:
			plans = pickle.load(f)
	
	folder = "/net/pasnas01/pool1/ramanha/corpus/nnUNet/nnUNet_preprocessed/Task003_Liver/nnUNetData_plans_v2.1_stage0"
	dataset = load_dataset(folder)
	unpack_dataset(folder)
	dataset_tr, dataset_val = do_split_custom(dataset, "normal")

	patch_size = get_patch_size([128, 128, 128], default_3D_augmentation_params["rotation_x"], default_3D_augmentation_params["rotation_y"], default_3D_augmentation_params["rotation_z"],default_3D_augmentation_params["scale_range"])
	
	dl_tr = DataLoader3D(dataset_tr, patch_size, np.array(plans['plans_per_stage'][0]['patch_size']).astype(int), 2)
	dl_val = DataLoader3D(dataset_val, np.array(plans['plans_per_stage'][0]['patch_size']).astype(int), np.array(plans['plans_per_stage'][0]['patch_size']).astype(int), 2)
		
	dl_tr = get_moreDA_augmentation(dataloader_train=dl_tr, patch_size=np.array(plans['plans_per_stage'][0]['patch_size']).astype(int))
	dl_val = get_moreDA_augmentation(dataloader_val=dl_val, patch_size=np.array(plans['plans_per_stage'][0]['patch_size']).astype(int))
	
	#metrics
	log = pd.DataFrame(index=[], columns=['epoch','lr','loss','train_iou','train_dice','val_loss','val_iou','val_dice'])
	best_dice=0
	trigger=0

	for epoch in range(config['epochs']):
		#train for 1 epoch
		train_log = train(dl_tr, model, criterion, optimizer)
		
		#evaluate on validation set
		val_log = validate(dl_val, model, criterion)
		val_scores = val_log[1][0]

		#Update Learning Rate		
		poly_lr(epoch)
		scheduler.step()

		print('Training epoch [{}/{}], Training loss:{:.4f}, Validation loss:{:.4f}, Validation DICE:{:.4f}, Validation IOU:{:.4f}'.format(
						epoch + 1,
						config['epochs'],
						train_log,
						val_log[0],
						val_scores["Mean Dice: \t"],
						val_scores["Mean IoU : \t"]))

		tmp = pd.Series([
			epoch,
			config['lr'],
			train_log,
			val_log[0],
			val_scores["Mean Dice: \t"],
			val_scores["Mean IoU : \t"]
		], index=['epoch','lr','loss','val_loss','val_iou','val_dice'])

		log = log.append(tmp, ignore_index=True)
		log.to_csv('model_outputs/{}/log.csv'.format(file_name), index=False)
		#log_running_metrics.to_csv('model_outputs/{}/log_running_metrics.csv'.format(file_name), index=False)

		trigger += 1

		if val_scores["Mean Dice: \t"] > best_dice:
			if torch.cuda.device_count() > 1:
				torch.save(model.module.state_dict(), 'model_outputs/{}_{}/best_model.pth'.format(file_name,config['loss']))
			else:
				torch.save(model.state_dict(), 'model_outputs/{}_{}/best_model.pth'.format(file_name,config['loss']))
			torch.save(optimizer.state_dict(), 'model_outputs/{}_{}/best_optimizer.pth'.format(file_name,config['loss']))
			best_dice = val_scores["Mean Dice: \t"]
			print("=> saved best model as validation DICE is greater than previous best DICE")
			trigger = 0

		#Save snapshot
		if torch.cuda.device_count() > 1:
			torch.save(model.module.state_dict(), 'model_outputs/{}_{}/last_model.pth'.format(file_name,config['loss']))
		else:
			torch.save(model.state_dict(), 'model_outputs/{}_{}/last_model.pth'.format(file_name,config['loss']))
		torch.save(optimizer.state_dict(), 'model_outputs/{}_{}/last_optimizer.pth'.format(file_name,config['loss']))

		#early stopping
		if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
			print("=> early stopping")
			break

		torch.cuda.empty_cache()

if __name__ == '__main__':
	main()
