import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import random
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

import segmentation_models_pytorch as smp

from tqdm import tqdm

class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Combines Convolution, Batch Norm, and ReLU into a single operation.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Executes the convolutional operation
        
        Args:
            x: input tensor

        Returns:
            Result of running the convolution operation on x.
        """
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Downsamples the output of a Convolution using MaxPool.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv = ConvBatchRelu(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Executes a convolution operation followed by MaxPool for downsampling.

        Args:
            x: input tensor

        Returns:
            down: Result of running the convolution operation.
            p: Result of running MaxPool2D on the output of the previous operation.
        """
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Deconvolution to upsample the input

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ConvBatchRelu(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Upscale the input tensor
        Concat the input x1 with x2
        Run Convolution on the resulting tensor

        Args:
            x1: input tensor
            x2: redidual tensor from the parralel layer on the encoder.
        
        Returns:
            A tensor representing the final result.
        """
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Basic Implementation of UNet. We use a reduced number of parameters to keep the size and training time
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 16)
        self.down_convolution_2 = DownSample(16, 32)
        self.down_convolution_3 = DownSample(32, 64)
        self.down_convolution_4 = DownSample(64, 128)

        self.bottle_neck = nn.Sequential(
            ConvBatchRelu(128, 256),   # Channel projection layer (3x3 kernel)
            ConvBatchRelu(256, 512),   # Spatial transition 1 (3x3 kernel)
            ConvBatchRelu(512, 256),   # Spatial transition 2 (3x3 kernel)
        )

        self.up_convolution_1 = UpSample(256, 128)
        self.up_convolution_2 = UpSample(128, 64)
        self.up_convolution_3 = UpSample(64, 32)
        self.up_convolution_4 = UpSample(32, 16)

        self.out = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out