import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    '''
    Inspiration from CBAM attention-module networks: https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py
    '''
    def __init__(self, num_channels, reduction_ratio=16, pool_types=['avg','max','std']):
        super(ChannelGate, self).__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(num_channels, num_channels // reduction_ratio), nn.ReLU(), nn.Linear(num_channels // reduction_ratio, num_channels))
        self.pool_types = pool_types
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type == 'max':
                max_pool = F.max_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type == 'std':
                std_pool = torch.std(x, dim=(2, 3,4))
                channel_att_raw = self.mlp( std_pool )
        scale = F.sigmoid( channel_att_raw ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def __init__(self, pool_types=['avg','max','std']):
        super(ChannelPool, self).__init__()
        self.pool_types = pool_types
    def forward(self, x):
        if not 'std' in self.pool_types:
            return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        else:
            return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1), torch.std(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self, pool_types=['avg','max','std']):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.pool_types = pool_types
        self.compress = ChannelPool(self.pool_types)
        self.spatial = BasicConv(len(self.pool_types), 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'std'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
