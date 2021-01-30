import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data

#Convolution blocks
    
class conv3DInstanceNorm(nn.Module): 
    def __init__(self,in_channels,n_filters,k_size,stride,padding,bias=True,dilation=1,is_InstanceNorm=True):
        super(conv3DInstanceNorm, self).__init__()

        conv_mod = nn.Conv3d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_InstanceNorm: 
            self.cb_unit = nn.Sequential(conv_mod, nn.InstanceNorm3d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs) 
        return outputs

class conv3DInstanceNormPRelu(nn.Module): #Defining own convolution with PReLU
    def __init__(self,in_channels,n_filters,k_size,stride,padding,bias=True,dilation=1,is_InstanceNorm=True,):
        super(conv3DInstanceNormPRelu, self).__init__()

        conv_mod = nn.Conv3d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=True,
            dilation=dilation,
        )

        if is_InstanceNorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.InstanceNorm3d(int(n_filters)), nn.PReLU(n_filters)#Prelu
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.PReLU(n_filters))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs) #CBRUnit: Conv, InstanceNorm, ReLU unit
        return outputs

class bottleNeckIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation=1, is_InstanceNorm=True):
        super(bottleNeckIdentity, self).__init__()

        bias = True

        self.cbr1 = conv3DInstanceNormPRelu(
            in_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_InstanceNorm=is_InstanceNorm
        )
        self.cb3 = conv3DInstanceNorm(
            out_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_InstanceNorm=is_InstanceNorm
        )
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        residual = x
        x = self.prelu (residual + self.cb3(self.cbr1(x)))
        return x


class residualBlock(nn.Module):
    def __init__(
        self,
        n_blocks,
        in_channels,
        out_channels,
        stride,
        dilation=1,
        include_range="all",
        is_InstanceNorm=True,
    ):
        super(residualBlock, self).__init__()

        if dilation > 1:
            stride = 1

        layers = []

        if include_range in ["all", "identity"]:
            for i in range(n_blocks):
                layers.append(
                    bottleNeckIdentity(
                        out_channels, in_channels, stride, dilation, is_InstanceNorm=is_InstanceNorm
                    )
                )

        self.layers = nn.Sequential(*layers) #add required number of layers to blocks

    def forward(self, x):
        return self.layers(x)


class cascadeFeatureFusion(nn.Module):
    def __init__(
        self, n_classes, low_in_channels, high_in_channels, out_channels, is_InstanceNorm=True
    ):
        super(cascadeFeatureFusion, self).__init__()

        bias = True #not is_InstanceNorm

        self.low_dilated_conv_bn = conv3DInstanceNorm(
            low_in_channels,
            out_channels,
            (3), 
            stride=(1,1,1),
            padding=(2),
            bias=bias,
            dilation=(2),
            is_InstanceNorm=is_InstanceNorm,
        )
        self.low_classifier_conv = nn.Conv3d(
            int(low_in_channels),
            int(n_classes),
            kernel_size=(3,5,5),
            padding=(1,2,2),
            stride=1,
            bias=True,
            dilation=1,
        )  # Train only
        self.high_proj_conv_bn = conv3DInstanceNorm(
            high_in_channels,
            out_channels,
            (3,5,5),
            stride=1,
            padding=(1,2,2),
            bias=bias,
            is_InstanceNorm=is_InstanceNorm,
        )
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x_low, x_high):
        x_low_upsampled = F.interpolate(
            x_low,(x_high.shape[2],x_high.shape[3],x_high.shape[4]), mode="trilinear", align_corners=True
        )           
        low_cls = self.low_classifier_conv(x_low_upsampled)
        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        high_fused_fm = self.prelu(low_fm + high_fm)

        return high_fused_fm, low_cls
