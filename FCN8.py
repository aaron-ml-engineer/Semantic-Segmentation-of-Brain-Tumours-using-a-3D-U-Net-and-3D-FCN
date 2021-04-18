import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as tf
from UNET3D import DoubleConvBlock

# Convolutional blocks used as the building blocks of FCN-8 architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False), # 3x3 kernel, stride 1, padding same
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class TripleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConvBlock, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channels), 
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channels), 
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channels), 
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)    
    
class FCN8(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[64, 128, 256, 512, 4096]): # out_channels would be 155 for different segmentation slices? or 4 for the tumour classes?
        super(FCN8, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=False)

        # Conv Block 1 - 2 conv layers
        self.conv1 = DoubleConvBlock(in_channels, out_channels=features[0])
        # Conv Block 2 - 2 conv layers
        self.conv2 = DoubleConvBlock(in_channels=features[0], out_channels=features[1])
        # Conv Block 3 - 3 layers
        self.conv3 = TripleConvBlock(in_channels=features[1], out_channels=features[2])
        # Conv Block 4 - 3 layers
        self.conv4 = TripleConvBlock(in_channels=features[2], out_channels=features[3])
        # Conv Block 5 - 3 layers
        self.conv5 = TripleConvBlock(in_channels=features[3], out_channels=features[3])
        # fc6 - single conv
        self.conv6= ConvBlock(in_channels=features[3], out_channels=features[4], kernel_size=7, padding=3)
        # fc7 - single conv 
        self.conv7= ConvBlock(in_channels=features[4], out_channels=features[4], kernel_size=1, padding=0) 
        # fc8 - single conv
        self.conv8= ConvBlock(in_channels=features[4], out_channels=out_channels, kernel_size=1, padding=0)

        # upsample prediction 
        self.upconv1= nn.ConvTranspose3d(in_channels=4, out_channels=features[3], kernel_size=4, stride=2, padding=1, output_padding=(1,0,0))
        self.upconv2= nn.ConvTranspose3d(in_channels=features[3], out_channels=features[2], kernel_size=4, stride=2, padding=1)
        self.upconv3= nn.ConvTranspose3d(in_channels=features[2], out_channels=out_channels, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x= self.pool(self.conv1(x)) 
        x= self.pool(self.conv2(x)) 
        x3= self.pool(self.conv3(x))
        x4=self.pool(self.conv4(x3))
        x= self.pool(self.conv5(x4))
        x= self.conv6(x)
        x= self.conv7(x)
        x= self.conv8(x)
        x= self.upconv1(x)
        x= x4 + x
        x= self.upconv2(x)
        x= x3 + x
        x= self.upconv3(x)
        return x