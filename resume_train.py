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
from models.vsnet_64ch import vsnet_64ch

from utils.metrics import runningScore
from utils.losses.crossentropy import categorical_cross_entropy
from utils.losses.focalloss import *
from data_loader.dataloader import *

from main import train, validate
	
def main():
	parser = argparse.ArgumentParser()
	
	#model
	parser.add_argument('model_path', type=str, metavar='N',
				help="model folder with config, model and log files")
	parser.add_argument('--early_stopping', default=40, type=int, metavar='N',
				help="early-stopping (default: 40)", dest='early_stopping')
				
	args = parser.parse_args()
	model_path = args.model_path
	
	folder_name = os.path.join('model_outputs',model_path)
	
	with open(os.path.join(folder_name, 'config.yml'), 'r') as stream:
		config = yaml.safe_load(stream)
	
	config['early_stopping'] = args.early_stopping
	
	print('-' * 20)
	print("Configuration settings as follows:")
	for key in config:
		print('{}: {}'.format(key, config[key]))
	print('-' * 20)
		
	#Criterion
	if config['loss'] == 'ce':
		criterion = categorical_cross_entropy(weight=config['alpha'])
	elif config['loss'] == 'fl':
		criterion = FocalLoss(alphas=config['alpha'], gamma=2.0)
	elif config['loss'] == 'reg_fl':
		criterion = RegulatedFocalLoss(alphas=[1.,1.,1.], gamma=2.0, weight=1e-4, modified=False) # Must add Gaussian
	else: print('Select a valid loss')

	cudnn.benchmark = True
	
	#Create model
	print("=> Loading model")
	if config['name'] == 'vsnet':
		model = vsnet()
	elif config['name'] == 'vsnet_with_attention':
		model = vsnet_with_attention()
	elif config['name'] == 'vsnet_depthwise':
		model = vsnet_depthwise()
	elif config['name'] == 'vsnet_64ch':
		model = vsnet_64ch()

	model = model.cuda()
	params = filter(lambda p: p.requires_grad, model.parameters())
	
        #Load last saved model
	print('Model path: {}'.format(os.path.join(folder_name, 'last_model.pt')))
	model.load_state_dict(torch.load(os.path.join(folder_name, 'last_model.pt')))
	
	#Optimizer
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
	
	optimizer.load_state_dict(torch.load(os.path.join(folder_name, 'last_optim.pt')))
	
	if torch.cuda.device_count() > 1:
		print("Let's use {} GPUs!".format(torch.cuda.device_count()))
		model = nn.DataParallel(model)
		
	#Load log file to find last epoch
	df = pd.read_csv(os.path.join(folder_name, 'log.csv'))
	last_epoch = list(df['epoch'])[-1]
	
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
	log = pd.DataFrame(index=[], columns=['epoch','lr','loss','val_loss','val_iou','val_dice'])
	best_dice=0
	trigger=0

	for epoch in range(last_epoch+1, config['epochs']):
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
		log.to_csv(os.path.join(folder_name, 'log2.csv'), index=False)
		#log_running_metrics.to_csv('model_outputs/{}_{}_{}_alpha_{}_{}_reg_{}_{}'.format(file_name,config['lr'],config['loss'],str(config['alpha'][0]),str(config['alpha'][1]),str(config['lr'],str(config['epochs']), index=False)

		trigger += 1

		if val_scores["Mean Dice: \t"] > best_dice:
			if torch.cuda.device_count() > 1:
				torch.save(model.module.state_dict(), os.path.join(folder_name, 'best_model.pt'))
			else:
				torch.save(model.state_dict(), os.path.join(folder_name, 'best_model.pt'))
			torch.save(optimizer.state_dict(), os.path.join(folder_name, 'best_optim.pt'))
			best_dice = val_scores["Mean Dice: \t"]
			print("=> saved best model as validation DICE is greater than previous best DICE")
			trigger = 0

		#Save snapshot
		if torch.cuda.device_count() > 1:
			torch.save(model.module.state_dict(), os.path.join(folder_name, 'last_model.pt'))
		else:
			torch.save(model.state_dict(), os.path.join(folder_name, 'last_model.pt'))
		torch.save(optimizer.state_dict(), os.path.join(folder_name, 'last_optim.pt'))

		#early stopping
		if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
			print("=> early stopping")
			break

		torch.cuda.empty_cache()
	
if __name__ == '__main__':
	main()

