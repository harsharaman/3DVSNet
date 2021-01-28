import torch
import torch.nn.functional as F

def loss_3d(input_, target, weight=None, size_average=True):
    n, c, d, h, w = input_.size()
    nt,ct, dt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt or d != dt:  # upsample labels
        input_ = F.interpolate(input_, size=(dt, ht, wt), mode="trilinear", align_corners=True)
    input_ = input_.transpose(1, 2).transpose(2, 3).transpose(3,4).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input_, target,reduction='mean'
    )
    return loss

def multi_scale_loss_3d(input_, target, weight=None, size_average=True, scale_weight=None):#[0.4,0.6,1][0.1,0.2,0.7]
    if not isinstance(input_, tuple):
        return loss_3d(input_=input_, target=target, weight=weight, size_average=size_average)

    # Auxiliary training 
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input_)
        scale = 0.25 #0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).cuda()
           # target.device
        
    scale_weight =  scale_weight/sum(scale_weight) # normalise
    loss = 0.0
    for i, inp in enumerate(input_):
        loss = (loss) + scale_weight[i] * loss_3d(
            input_=inp, target=target, weight=weight, size_average=True
        )
    return loss
