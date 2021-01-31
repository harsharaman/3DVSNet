import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
sys.path.append(os.path.join(os.getcwd(), 'models'))
from vsnet_parts_separable import convDepthwiseInstanceNorm, convDepthwiseInstanceNormPRelu, bottleNeckIdentity, residualBlock, cascadeFeatureFusion
from SEblocks import ChannelSELayer3D, SpatialSELayer3D, ChannelSpatialSELayer3D

class vsnet_depthwise(nn.Module):
    def __init__(
        self,
        n_classes=3,
        block_config=[1, 1, 2, 1],
        is_InstanceNorm=True,
    ):

        super(vsnet_depthwise, self).__init__()

        bias = True 
        self.block_config = block_config
        self.n_classes =  n_classes

        # Encoder
        self.convbnrelu1_1 = convDepthwiseInstanceNormPRelu(
            in_channels=1,
            k_size=(3,5,5),
            n_filters=32,
            padding=(1,2,2),
            stride=(1,2,2),
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )
        
        self.scse26 = ChannelSELayer3D(32)
        
        self.convbnrelu1_2 = convDepthwiseInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),
            n_filters=32,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )
        
        self.convbnrelu1_3 = convDepthwiseInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),
            n_filters=32,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )

        self.res_block3_identity = residualBlock(
            self.block_config[1], 
            32,
            32,
            2,
            1,
            include_range="identity",
            is_InstanceNorm=is_InstanceNorm,
        )

        # Final conv layer in LR branch
        self.conv5_4_k1 = convDepthwiseInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),
            n_filters=32,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )

        # High-resolution (sub1) branch
        self.convbnrelu1_sub1 = convDepthwiseInstanceNormPRelu(
            in_channels=1,
            k_size=(3,5,5),
            n_filters=16,
            padding=(1,2,2),
            stride=(1,2,2),
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )
        self.scse13 = ChannelSELayer3D(16)
                                             
        self.convbnrelu2_sub1 = convDepthwiseInstanceNormPRelu(
            in_channels=16,
            k_size=(3,5,5),
            n_filters=32,
            padding=(1,2,2),
            stride=(1,2,2),
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )

        self.convbnrelu1_3 = convDepthwiseInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),
            n_filters=32,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )

        self.convbnrelu4_sub1 = convDepthwiseInstanceNormPRelu(
            in_channels=16,
            k_size=(3,5,5),
            n_filters=16,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )
        self.convbnrelu3_sub1 = convDepthwiseInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),
            n_filters=16,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )
       
        self.convbnrelu5_sub1 = convDepthwiseInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),
            n_filters=16,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )
        
        self.classification = nn.Conv3d(16, self.n_classes, 1)

        # Cascade Feature Fusion Units

        self.cff_sub12 = cascadeFeatureFusion(
            self.n_classes, 32, 32, 32, is_InstanceNorm=is_InstanceNorm
        )

    def forward(self, x):
        d, h, w = x.shape[2:]
        
        # Low resolution branch
        x_sub2 = F.interpolate(
            x, size=(int(d), int(h/2), int(w/2)), mode="trilinear", align_corners=True
        )
        x_sub2 = self.convbnrelu1_1(x_sub2)
        x_sub2 = self.convbnrelu1_2(x_sub2)
        x_sub2 = self.convbnrelu1_3(x_sub2)
        x_sub2 = F.max_pool3d(x_sub2, 3, 2, 1) 
        x_sub2 = self.res_block3_identity(x_sub2)
        x_sub2 = self.conv5_4_k1(x_sub2)
        
        # High resolution branch
        x_sub1 = self.convbnrelu1_sub1(x)
        syn1 = x_sub1
        x_sub1 = self.convbnrelu2_sub1(x_sub1)

        # fusion
        x_sub12, sub2_cls = self.cff_sub12(x_sub2, x_sub1)

        x_sub12 = F.interpolate(
            x_sub12, size=(int(d), int(h/2), int(w/2)), mode="trilinear", align_corners=True
        )
        x_sub12 = self.convbnrelu5_sub1(x_sub12)
        concat = torch.cat((x_sub12, syn1), dim=1)
        x_sub12 = self.convbnrelu3_sub1(concat)
        x_sub12 = self.convbnrelu4_sub1(x_sub12)
        sub124_cls = self.classification(x_sub12)

        if self.training:
            return (sub124_cls, sub2_cls)
        else:
            sub124_cls = F.interpolate(
                sub124_cls,
                size=(d,h,w),
                mode="trilinear",
                align_corners=True,
            )
            return sub124_cls


#test model
input = torch.randn(2,1,32,32,32).cuda()
net = vsnet_depthwise().float().cuda()
#net = nn.DataParallel(net)
print(net)
output=net(input)
print(output[0].shape)


