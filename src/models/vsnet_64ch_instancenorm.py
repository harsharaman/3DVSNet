import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data

#Defining all building blocks
class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
       
        squeeze_tensor = self.avg_pool(input_tensor)
        #squeeze_tensor_std = torch.std(input_tensor)
        #squeeze_tensor = squeeze_tensor + squeeze_tensor_std
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor
    
class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor
    


class ChannelSpatialSELayer3D(nn.Module):
       """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """
       def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)
       def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
    
class conv3DInstanceNorm(nn.Module): #Defining own convolution with needed parameters because different convolutions needed
    def __init__(self,in_channels,n_filters,k_size,stride,padding,bias=True,dilation=1,is_instancenorm=True):
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

        if is_instancenorm: 
            self.cb_unit = nn.Sequential(conv_mod, nn.InstanceNorm3d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs) #CBUnit: Conv, BatchNorm unit
        return outputs

class conv3DInstanceNormPRelu(nn.Module): #Defining own convolution with PReLU
    def __init__(self,in_channels,n_filters,k_size,stride,padding,bias=True,dilation=1,is_instancenorm=True,):
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

        if is_instancenorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.InstanceNorm3d(int(n_filters)), nn.PReLU(n_filters)#Prelu
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.PReLU(n_filters))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs) #CBRUnit: Conv, BatchNorm, ReLU unit
        return outputs

class bottleNeckIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation=1, is_instancenorm=True):
        super(bottleNeckIdentity, self).__init__()

        bias = True

        self.cbr1 = conv3DInstanceNormPRelu(
            in_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_instancenorm=is_instancenorm
        )
        self.cb3 = conv3DInstanceNorm(
            out_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_instancenorm=is_instancenorm
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
        is_instancenorm=True,
    ):
        super(residualBlock, self).__init__()

        if dilation > 1:
            stride = 1

        layers = []

        if include_range in ["all", "identity"]:
            for i in range(n_blocks):
                layers.append(
                    bottleNeckIdentity(
                        out_channels, in_channels, stride, dilation, is_instancenorm=is_instancenorm
                    )
                )

        self.layers = nn.Sequential(*layers) #add required number of layers to blocks

    def forward(self, x):
        return self.layers(x)


class cascadeFeatureFusion(nn.Module):
    def __init__(
        self, n_classes, low_in_channels, high_in_channels, out_channels, is_instancenorm=True
    ):
        super(cascadeFeatureFusion, self).__init__()

        bias = True #not is_instancenorm

        self.low_dilated_conv_bn = conv3DInstanceNorm(
            low_in_channels,
            out_channels,
            (3), 
            stride=(1,1,1),
            padding=(2),
            bias=bias,
            dilation=(2),
            is_instancenorm=is_instancenorm,
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
            is_instancenorm=is_instancenorm,
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


def get_interp_size(input, s_factor=1, z_factor=1):  # for caffe (function not used)

    ori_d, ori_h, ori_w = input.shape[2:]
    
    # shrink (s_factor >= 1)
    ori_d = (ori_d -1) / s_factor + 1 
    ori_h = (ori_h -1) / s_factor + 1 
    ori_w = (ori_w -1) / s_factor + 1

    # zoom (z_factor >= 1)
    ori_d = (ori_d) + (ori_d -1 ) * (z_factor - 1)
    ori_h = (ori_h) + (ori_h -1 ) * (z_factor - 1)
    ori_w = (ori_w)+ (ori_w -1 ) * (z_factor - 1)

    resize_shape = (int(ori_d), int(ori_h), int(ori_w))

    return resize_shape


# In[38]:


class vsnet(nn.Module):
    def __init__(
        self,
        n_classes=3,
        block_config=[1, 1, 2, 1],#what is block_config?
        is_instancenorm=False,
    ):

        super(vsnet, self).__init__()

        bias = True 
        self.block_config = block_config
        self.n_classes =  n_classes

        # Encoder
        self.convbnrelu1_1 = conv3DInstanceNormPRelu(
            in_channels=1,
            k_size=(3,5,5),#3
            n_filters=64,
            padding=(1,2,2),
            stride=(1,2,2),
            bias=bias,
            is_instancenorm=is_instancenorm,
        )
        
        self.scse26 = ChannelSpatialSELayer3D(64)
        
        self.convbnrelu1_2 = conv3DInstanceNormPRelu(
            in_channels=64,
            k_size=(3,5,5),#3
            n_filters=64,
            padding=(1,2,2),#1
            stride=1,
            bias=bias,
            is_instancenorm=is_instancenorm,
        )
        
        self.convbnrelu1_3 = conv3DInstanceNormPRelu(
            in_channels=64,
            k_size=(3,5,5),#3
            n_filters=64,
            padding=(1,2,2),#1
            stride=1,
            bias=bias,
            is_instancenorm=is_instancenorm,
        )

        self.res_block3_identity = residualBlock(
            self.block_config[1], #block config used here
            64,
            64,
            2,
            1,
            include_range="identity",
            is_instancenorm=is_instancenorm,
        )

        # Final conv layer in LR branch
        self.conv5_4_k1 = conv3DInstanceNormPRelu(
            in_channels=64,
            k_size=(3,5,5),
            n_filters=64,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_instancenorm=is_instancenorm,
        )

        # High-resolution (sub1) branch
        self.convbnrelu1_sub1 = conv3DInstanceNormPRelu(
            in_channels=1,
            k_size=(3,5,5),#3
            n_filters=32,
            padding=(1,2,2),
            stride=(1,2,2),
            bias=bias,
            is_instancenorm=is_instancenorm,
        )
        self.scse13 = ChannelSpatialSELayer3D(32)
                                             
        self.convbnrelu2_sub1 = conv3DInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),#3
            n_filters=64,
            padding=(1,2,2),
            stride=(1,2,2),
            bias=bias,
            is_instancenorm=is_instancenorm,
        )

        self.convbnrelu1_3 = conv3DInstanceNormPRelu(
            in_channels=64,
            k_size=(3,5,5),#3
            n_filters=64,
            padding=(1,2,2),#1
            stride=1,
            bias=bias,
            is_instancenorm=is_instancenorm,
        )

        self.convbnrelu4_sub1 = conv3DInstanceNormPRelu(
            in_channels=32,
            k_size=(3,5,5),#3
            n_filters=32,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_instancenorm=is_instancenorm,
        )
        self.convbnrelu3_sub1 = conv3DInstanceNormPRelu(
            in_channels=64,
            k_size=(3,5,5),#3
            n_filters=32,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_instancenorm=is_instancenorm,
        )
       
        self.convbnrelu5_sub1 = conv3DInstanceNormPRelu(
            in_channels=64,
            k_size=(3,5,5),#3
            n_filters=32,
            padding=(1,2,2),
            stride=1,
            bias=bias,
            is_instancenorm=is_instancenorm,
        )
        
        self.classification = nn.Conv3d(32, self.n_classes, 1)

        # Cascade Feature Fusion Units

        self.cff_sub12 = cascadeFeatureFusion(
            self.n_classes, 64, 64, 64, is_instancenorm=is_instancenorm
        )

    def forward(self, x):
        d, h, w = x.shape[2:]
        
        # Low resolution branch
        x_sub2 = F.interpolate(
            x, size=(int(d), int(h/2), int(w/2)), mode="trilinear", align_corners=True
        )
        x_sub2 = self.convbnrelu1_1(x_sub2)
        x_sub2 = self.scse26(x_sub2)
        x_sub2 = self.convbnrelu1_2(x_sub2)
        x_sub2 = self.scse26(x_sub2)
        x_sub2 = self.convbnrelu1_3(x_sub2)
        x_sub2 = self.scse26(x_sub2)
        x_sub2 = F.max_pool3d(x_sub2, 3, 2, 1) 
        x_sub2 = self.res_block3_identity(x_sub2)
        x_sub2 = self.conv5_4_k1(x_sub2)
        x_sub2 = self.scse26(x_sub2)
        
        # High resolution branch
        x_sub1 = self.convbnrelu1_sub1(x)
        x_sub1 = self.scse13(x_sub1)
        syn1 = x_sub1
        x_sub1 = self.convbnrelu2_sub1(x_sub1)
        x_sub1 = self.scse26(x_sub1)
        
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
net = vsnet().float().cuda()
#net = nn.DataParallel(net)
print(net)
output=net(input)
print(output[0].shape)

