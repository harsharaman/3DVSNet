import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Entire code directly taken from Trapthi's work
'''

def loss_3d(input_, target, weight=None):
    n, c, d, h, w = input_.size()
    nt,ct, dt, ht, wt = target.size()
    weight = torch.FloatTensor(weight).cuda()
    # Handle inconsistent size between input and target
    if h != ht or w != wt or d != dt:  # upsample labels
        input_ = F.interpolate(input_, size=(dt, ht, wt), mode="trilinear", align_corners=True)
    input_ = input_.transpose(1, 2).transpose(2, 3).transpose(3,4).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input_, target,reduction='mean', weight=weight
    )
    return loss

def multi_scale_loss_3d(input_, target, weight=None, scale_weight=None):
    if not isinstance(input_, tuple):
        return loss_3d(input_=input_, target=target, weight=weight)

    # Auxiliary training 
    if scale_weight is None:  
        n_inp = len(input_)
        scale = 0.25 
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).cuda() #weight for input from HR and LR branches
           
    scale_weight =  scale_weight/sum(scale_weight) # normalise
    loss = 0.0
    for i, inp in enumerate(input_):
        loss = (loss) + scale_weight[i] * loss_3d(
            input_=inp, target=target, weight=weight
        )
    return loss

class categorical_cross_entropy(nn.Module):
    '''
    Class wrapper to multi_scale_loss_3d
    '''
    def __init__(self, weight=None, scale_weight=None):
        super(categorical_cross_entropy, self).__init__()
        self.weight = weight
        self.scale_weight = scale_weight

    def forward(self, input_, target):
        return multi_scale_loss_3d(input_, target, self.weight, self.scale_weight)
