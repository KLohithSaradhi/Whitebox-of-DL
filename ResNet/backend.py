import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        # Flag if the Block does (Spatial downsampling and feature map doubling) or not
        self.downsample = downsample   

        if self.downsample:

            # Conv for Spatial Downsampling
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 2, padding=1)

            # Conv for increasing feature maps in "X" when adding the residual to the "out"
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2)
        else:

            # else default Convolution
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


        # Second Convolution that transforms in with the same dimensions retained
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        #activation
        self.relu = nn.ReLU()

    def forward(self, X):

        # Conventional CNN
        out = self.conv1(X)
        out = self.relu(out)
        out = self.conv2(out)

        # Residual Connection
        if self.downsample:
            # We increase feature maps and reduce spatial size to match dimensions
            X = self.downsample_conv(X)


        out = X + out

        out = self.relu(X)

        return out
    

class ResNet(nn.Module):

    def __init__(self, config, in_channels):
        '''
        config format:
            {feature_maps : count, ...}
        '''

        self.features_maps = self.config.keys
        self.count = self.config.values

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.features_maps[0], kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d()

        self.blocks = nn.Sequential()


