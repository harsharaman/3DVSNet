from typing import Optional, List, Union

import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
import numpy as np

#Based on Kornia code for Focal Loss: 
#See https://github.com/kornia/kornia/issues/551

def focal_loss(input_, target, alphas, fl_type:str, gamma=2, reduction='none', modified=False):
	
	if fl_type not in ['binary', 'multiclass', 'multilabel']: #We use Multi-Class, and only mutli-class is properly implemented
		raise ValueError(f"fl_type should be binary, multiclass or multilabel instead of {fl_type}.")

	if fl_type == 'multiclass':
		
		#If the target is double the spatial size of input (during training), we interpolate them to a proper size
		n, c, d, h, w = input_.size()
		nt,ct, dt, ht, wt = target.size()

		# Handle inconsistent size between input and target
		if h != ht or w != wt or d != dt:  # upsample labels
			input_ = F.interpolate(input_, size=(dt, ht, wt), mode="trilinear", align_corners=True)

		if input_.shape[0] != target.shape[0]:
			raise ValueError(f'Expected input batch_size ({input_.shape[0]}) to match target batch_size ({target.shape[0]}).')

		if target.max() >= input_.shape[1]:
			raise ValueError(f"There are more target classes ({target.max()+1}) than the number of classes predicted ({input_.shape[1]})")

	if not input_.device == target.device:
		raise ValueError("input and target must be in the same device. Got: {} and {}" .format(input_.device, target.device))

	if gamma < 1:
		raise RuntimeError('Backpropagation problems. See EfficientDet Rwightman focal loss implementation')

	# Create an alphas tensor. Move it to the same device and have the same dtype that the inputs

	if fl_type == 'binary' and (not isinstance(alphas, torch.Tensor) or alpha.ndim == 0): 
		if not 0 < alphas < 1: 
			raise ValueError(f"Alpha must be between 0 and 1 and it's {alphas}.")
		alphas = torch.tensor([alphas, 1-alphas], dtype=input.dtype, device=input.device) 
	elif isinstance(alphas, (tuple, list)): 
		alphas = torch.tensor(alphas, dtype=input_.dtype, device=input_.device)
	elif isinstance(alphas, torch.Tensor): 
		alphas = alphas.type_as(input_).to(input_.device)
	else:
		raise RuntimeError(f"Incorrect alphas type: {type(alphas)}. Alphas values {alphas}")

	# Normalize alphas to sum up to 1
	alphas.div_(alphas.sum()) #div_() is in-place version of div()

	# Non weighted version of Focal Loss computation:
	if fl_type == 'multiclass':
		input_ = input_.transpose(1, 2).transpose(2, 3).transpose(3,4).contiguous().view(-1, c)
		target = target.view(-1).long()
		base_loss = F.cross_entropy(input_, target, reduction='none')
	else:
		base_loss = F.binary_cross_entropy_with_logits(input_, target, reduction='none')

	target = target.type(torch.long)
	at = alphas.gather(0, target.data.view(-1)) #Assigns the respective alpha value to each class
	pt = torch.exp(-base_loss)
	
	if modified:
		focal_loss = at*(((1-pt)/pt)** gamma) * base_loss
	else:
		focal_loss = at*((1-pt)** gamma) * base_loss #we do not add a negative term becuase base_loss = - log(pt), and so two negatives cancel each other out
		
	if reduction == 'none': return focal_loss
	elif reduction == 'mean': return focal_loss.mean()
	elif reduction == 'sum': return focal_loss.sum()
	else: raise NotImplementedError("Invalid reduction mode: {}".format(reduction))

class FocalLoss(nn.Module):
	"""	
	Focal loss that support multiclass classification. See [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

	According to the paper, the Focal Loss for binary case is computed as follows:
	.. math::
		\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
		
	where:
	   - :math:`p_t` is the model's estimated probability for each class.
	
	The `input` is expected to contain raw, unnormalized scores for each class. `input` has to be a Tensor of size 
	either :math:`(minibatch, C)` for multilabel or multiclass classification or :math:`(minibatch, )` for binary classification. 
	
	The `target` is expected to contain raw, unnormalized scores for each class. `target` has to be a one hot encoded Tensor of size 
	either :math:`(minibatch, C)` for multilabel classification or :math:`(minibatch, )` for binary or multiclass classification. 
	
	Args:
		alphas (float, list, tuple, Tensor): the `alpha` value for each class. It weights the losses of each class. When `fl_type`
			is 'binary', it could be a float. In this case, it's transformed to :math:`alphas = (alphas, 1 - alphas)` where the
			first position is for the negative class and the second the positive. Note: alpha values are normalized to sum up 1.
		gamma (float): gamma exponent of the focal loss. Typically, between 0.25 and 4.
		reduction (string, optional): Specifies the reduction to apply to the output:
			``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
			``'mean'``: the sum of the output will be divided by the number of
			elements in the output, ``'sum'``: the output will be summed.
			
	
	:math:`(minibatch, C, d_1, d_2, ..., d_K)`
	
	Note: the implementation is based on [this post](https://amaarora.github.io/2020/06/29/FocalLoss.html).
	
	"""
	def __init__(self, alphas: Union[float, List[float]], fl_type: str="multiclass", gamma: float = 2.0, reduction: str = 'mean', modified=False) -> None:
		super(FocalLoss, self).__init__()
		self.fl_type = fl_type
		self.alphas = alphas 		
		self.gamma = gamma
		self.reduction = reduction
		self.modified = modified
	def forward(self, input_, target):
		if not isinstance(input_, tuple): #For validation
			return focal_loss(input_, target, fl_type=self.fl_type, alphas=self.alphas, gamma=self.gamma, reduction=self.reduction)
		else:
			loss=0.0
			for i, inp in enumerate(input_): #For training
				loss += focal_loss(inp, target, fl_type=self.fl_type, alphas=self.alphas, gamma=self.gamma, reduction=self.reduction)
			return loss

class RegulatedFocalLoss(FocalLoss):
	"""	
	Focal loss with Regularization that support multiclass classification. 

	Regularization term is: (L1-measure of (Gradient of GT label - Gradient of Estimated Label)) / (L1-measure of GT label + L1-measure of Estimated Label + Stability_term)
	
	Args:
		alphas (float, list, tuple, Tensor): the `alpha` value for each class. It weights the losses of each class. When `fl_type`
			is 'binary', it could be a float. In this case, it's transformed to :math:`alphas = (alphas, 1 - alphas)` where the
			first position is for the negative class and the second the positive. Note: alpha values are normalized to sum up 1.
		gamma (float): gamma exponent of the focal loss. Typically, between 0.25 and 4.
		reduction (string, optional): Specifies the reduction to apply to the output:
			``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
			``'mean'``: the sum of the output will be divided by the number of
			elements in the output, ``'sum'``: the output will be summed.
		epsilon (float): Stability term added to the denominator to prevent zero-division. Has no impact on the overall calculated value.
		weight (float): Weight given to the regularizing term in the overall loss
	
	:math:`(minibatch, C, d_1, d_2, ..., d_K)`
	
	Note: the implementation is based on [this post](https://amaarora.github.io/2020/06/29/FocalLoss.html).
	
	"""
	def __init__(self, alphas: Union[float, List[float]], fl_type: str="multiclass", gamma: float = 2.0, reduction: str = 'mean', modified=False, epsilon=1e-6, weight=0.25):
		super(RegulatedFocalLoss, self).__init__(alphas=alphas, fl_type=fl_type, gamma=gamma, reduction=reduction, modified=modified)
		self.alphas = alphas
		self.fl_type = fl_type
		self.alphas = alphas 
		self.gamma = gamma
		self.reduction = reduction
		self.modified = modified
		self.weight = weight
		self.epsilon = epsilon
		
	def forward(self, input_, target):
		loss = super().forward(input_, target) #Focal loss
		if isinstance(input_, tuple):	
			n, c, d, h, w = input_[0].size()
		else:
			n, c, d, h, w = input_.size()
		nt,ct, dt, ht, wt = target.size()
		# Handle inconsistent size between input and target
		if h != ht or w != wt or d != dt:  # upsample labels
			input_ = F.interpolate(input_[0], size=(dt, ht, wt), mode="trilinear", align_corners=True)
		
		# Taking gradient
		target_x, target_y, target_z = np.gradient(target.cpu().numpy(), axis=(2,3,4))
		input_pred = input_.data.max(1)[1].unsqueeze(1).cpu().numpy() #Estimating labels from input probablities
		input_x, input_y, input_z = np.gradient(input_pred, axis=(2,3,4))
		target_grad = np.sqrt(np.square(target_x) + np.square(target_y) + np.square(target_z))
		input_grad = np.sqrt(np.square(input_x) + np.square(input_y) + np.square(input_z))	
		
		#Calculating L1-norm, first for height and width dimensions, then for depth, and finally along (dummy) channel axis
		numerator = np.linalg.norm((np.linalg.norm((np.linalg.norm((target_grad-input_grad), ord=1, axis=(3,4))), ord=1, axis=2)), ord=1, axis=1)
		target_l1 = np.linalg.norm((np.linalg.norm((np.linalg.norm(target_grad, ord=1, axis=(3,4))), ord=1, axis=2)), ord=1, axis=1)
		input_l1 = np.linalg.norm((np.linalg.norm((np.linalg.norm(input_grad, ord=1, axis=(3,4))), ord=1, axis=2)), ord=1, axis=1)
		denominator = target_l1 + input_l1 + self.epsilon
			
		regulation = torch.tensor((numerator / denominator).sum()) #Sum() to sum over all instances in a batch
		#print("Regulation: {}".format(regulation))
		return loss + self.weight * regulation

class RegulatedFocalLossGaussianBlur(FocalLoss):
	def __init__(self, alphas: Union[float, List[float]], fl_type: str="multiclass", gamma: float = 2.0, reduction: str = 'mean', modified=False, epsilon=1e-6, weight=0.25, before_gradient=False):
		'''
		Same as Regulated Focal Loss, but with Gaussian blur added for labels
		super().__init__(alphas=alphas, fl_type=fl_type, gamma=gamma, reduction=reduction, modified=modified)
		Args:
		before_gradient (boolean): If this is set to true, Gaussian blur is applied before the gradient of labels is computed. 
		Otherwise, gradient of labels is computed first, and then the blur is applied
		'''
		self.alphas = alphas
		self.fl_type = fl_type
		self.alphas = alphas 		
		self.gamma = gamma
		self.reduction = reduction
		self.modified = modified
		self.weight = weight
		self.epsilon = epsilon
		self.before_gradient = before_gradient

	def forward(self, input_, target):
		loss = super().forward(input_, target) #Focal loss
		if isinstance(input_, tuple):	
			n, c, d, h, w = input_[0].size()
		else:
			n, c, d, h, w = input_.size()
		nt,ct, dt, ht, wt = target.size()
		# Handle inconsistent size between input and target
		if h != ht or w != wt or d != dt:  # upsample labels
			input_ = F.interpolate(input_[0], size=(dt, ht, wt), mode="trilinear", align_corners=True)
		if self.before_gradient:
			#Applying Gaussian blur
			target = gaussian_filter(target.cpu().numpy(),1)
			input_ = gaussian_filter(input_.data.max(1)[1].unsqueeze(1).cpu().numpy(),1)		
			# Taking gradient
			target_x, target_y, target_z = np.gradient(target, axis=(2,3,4))
			input_x, input_y, input_z = np.gradient(input_, axis=(2,3,4))
			target_grad = np.sqrt(np.square(target_x) + np.square(target_y) + np.square(target_z))
			input_grad = np.sqrt(np.square(input_x) + np.square(input_y) + np.square(input_z))	
		else:
			# Taking gradient
			target_x, target_y, target_z = np.gradient(target.cpu().numpy(), axis=(2,3,4))
			input_pred = input_.data.max(1)[1].unsqueeze(1).cpu().numpy()
			input_x, input_y, input_z = np.gradient(input_pred, axis=(2,3,4))
			target_grad = np.sqrt(np.square(target_x) + np.square(target_y) + np.square(target_z))
			input_grad = np.sqrt(np.square(input_x) + np.square(input_y) + np.square(input_z))
			#Applying Gaussian blur
			target_grad = gaussian_filter(target_grad,1)
			input_grad= gaussian_filter(input_grad,1)

		#Calculating L1-norm, first for height and width dimensions, then for depth, and finally along (dummy) channel axis
		numerator = np.linalg.norm((np.linalg.norm((np.linalg.norm((target_grad-input_grad), ord=1, axis=(3,4))), ord=1, axis=2)), ord=1, axis=1)
		target_l1 = np.linalg.norm((np.linalg.norm((np.linalg.norm(target_grad, ord=1, axis=(3,4))), ord=1, axis=2)), ord=1, axis=1)
		input_l1 = np.linalg.norm((np.linalg.norm((np.linalg.norm(input_grad, ord=1, axis=(3,4))), ord=1, axis=2)), ord=1, axis=1)
		denominator = target_l1 + input_l1 + self.epsilon
			
		regulation = torch.tensor((numerator / denominator).sum()) #Sum() to sum over all instances in a batch
		return loss + self.weight * regulation


