import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from typing import List
import copy
from torchvision.models.efficientnet import MBConvConfig,MBConv
from torchvision.ops.misc import ConvNormActivation


class ASPP(nn.Module):
    def __init__(self, num_classes, concat=True, return_features=False):
        super(ASPP, self).__init__()
        self.concat = concat
        self.return_features = return_features

        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        padding = 1
        output_padding = 1

        self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) 
        self.upsample_2 = self.upsample(256+64, 256, 3, padding, output_padding) 

        self.conv_1x1_4 = nn.Conv2d(256 + 64, num_classes, kernel_size=1, padding=0)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, x_128=None, x_64=None, x_32=None, x_16=None):

        feature_map_h = x_high_feature.size()[2]
        feature_map_w = x_high_feature.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) 
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) 
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) 
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) 

        out_img = self.avg_pool(x_high_feature) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 

        x = self.upsample_1(out)

        x = torch.cat([x, x_64], 1)
        x = self.upsample_2(x)
        neck_out_features = x
        x = self.conv_1x1_4(torch.cat([x, x_128], 1)) 
        if self.return_features:
            return x, neck_out_features

        return x

class ASPP_Efficientnet_upsampled(nn.Module):
    def __init__(self, num_classes, return_features=False):
        super(ASPP_Efficientnet_upsampled, self).__init__()
        self.return_features = return_features
        self.conv_1x1_1 = nn.Conv2d(448, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)
        self.conv_3x3_1 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)
        self.conv_3x3_2 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)
        self.conv_3x3_3 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(448, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)
        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
        padding = 1
        output_padding = 1
        self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
        self.upsample_2 = self.upsample(256+32, 256, 3, padding, output_padding)
        self.conv_1x1_4 = nn.Conv2d(256 + 24, num_classes, kernel_size=1, padding=0)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, l3=None,l2=None):
        feature_map_h = x_high_feature.size()[2]
        feature_map_w = x_high_feature.size()[3]
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) 
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) 
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) 

        out_img = self.avg_pool(x_high_feature) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        x = self.upsample_1(out)
        x = torch.cat([x, l3], 1)
        x = self.upsample_2(x)
        neck_out_features = x
        x = self.conv_1x1_4(torch.cat([x, l2], 1)) 
        if self.return_features:
            return x, neck_out_features
        return x

class efficientnet_upsampled(nn.Module):
    def __init__(self, input_channels=3):
        super(efficientnet_upsampled,self).__init__()
        print("efficientnet_b4")
        efficientnet = models.efficientnet_b4()
        old_conv1 = efficientnet.features[0][0]
        new_conv1 = nn.Conv2d(
            in_channels=input_channels,  
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=True if old_conv1.bias else False,
        )
        new_conv1.weight[:, :old_conv1.in_channels, :, :].data.copy_(old_conv1.weight.clone())
        efficientnet.features[0][0] = new_conv1
        self.efficientnet = nn.Sequential(*list(efficientnet.children())[0])
        block = MBConv
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: List[nn.Module] = []
        width_mult = 1.4
        depth_mult=1.8
        stochastic_depth_prob = 0.2
        bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
        inverted_residual_setting = [
                            bneck_conf(6, 3, 1, 40, 80, 3),
                            bneck_conf(6, 5, 1, 80, 112, 3),
                            bneck_conf(6, 5, 1, 112, 192, 4),
                            bneck_conf(6, 3, 1, 192, 320, 1),
                            ]
        self.eff_layer_2 = nn.Sequential(*list(self.efficientnet.children())[:2])
        self.eff_layer_3 = nn.Sequential(*list(self.efficientnet.children())[2:3])
        self.eff_layer_4 = nn.Sequential(*list(self.efficientnet.children())[3:4])
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels =  lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                    norm_layer=norm_layer, activation_layer=nn.SiLU))
        self.final_layer = nn.Sequential(*layers)


    def forward(self,x):
        l2 = self.eff_layer_2(x)
        l3 = self.eff_layer_3(l2)
        l4 = self.eff_layer_4(l3)
        final = self.final_layer(l4)
        return final,l3,l2

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) 

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(x)

        out = F.relu(out) 

        return out

class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, input_channels = 3):
        super(ResNet_BasicBlock_OS8, self).__init__()
        resnet = models.resnet34(pretrained=True)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-4])
        # first conv, bn, relu
        self.resnet_layer_1 = nn.Sequential(*list(resnet.children())[:-7]) 
        # max pooling, resnet block
        self.resnet_layer_2 = nn.Sequential(*list(resnet.children())[-7:-5]) 
        # resnet block
        self.resnet_layer_3 = nn.Sequential(*list(resnet.children())[-5:-4])
        num_blocks_layer_4 = 6
        num_blocks_layer_5 = 3
        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)
        print ("resnet 34")

    def forward(self, x):
        x_128 = self.resnet_layer_1(x)
        x_64 = self.resnet_layer_2(x_128)
        x_32 = self.resnet_layer_3(x_64)
        x_16 = self.layer4(x_32)
        x_high_feature = self.layer5(x_16)
        return x_high_feature, x_128, x_64, x_32, x_16

class DeepLabV3(nn.Module):
    def __init__(self, num_classes, efficientnet_key:bool=False, input_channels=3, return_features=False):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes
        self.efficientnet_key = efficientnet_key
        self.return_features = return_features

        if not efficientnet_key:
            self.resnet = ResNet_BasicBlock_OS8(input_channels=input_channels) 
            self.aspp = ASPP(num_classes=self.num_classes, return_features=self.return_features) 
        else:
            self.efficientnet = efficientnet_upsampled(input_channels=input_channels)
            self.aspp = ASPP_Efficientnet_upsampled(num_classes=self.num_classes, return_features=self.return_features) 

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        if self.efficientnet_key == None:
            x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
            output, neck_out_features = self.aspp(x_high_feature, x_128, x_64, x_32, x_16)
        else:
            l9,l3,l2 = self.efficientnet(x)
            output, neck_out_features = self.aspp(l9,l3,l2)
        mask,binary_code = torch.split(output,[1,self.num_classes-1],1)
        if self.return_features:
            return mask, binary_code, neck_out_features
        return mask, binary_code