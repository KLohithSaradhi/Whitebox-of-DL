import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        # Flag if the Block does (Spatial downsampling and feature map doubling) or not
        self.downsample = downsample   

        if self.downsample > 1:

            # Conv for Spatial Downsampling
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = downsample, padding=1)

            # Conv for increasing feature maps in "X" when adding the residual to the "out"
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = downsample)
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
        if self.downsample > 1:
            # We increase feature maps and reduce spatial size to match dimensions
            X = self.downsample_conv(X)


        out = X + out

        out = self.relu(X)

        return out
    

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, downsample):
        super().__init__()

        # Flag if the Block does (Spatial downsampling and feature map doubling) or not
        self.downsample = downsample   

        self.in_bottleneck = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size = 1)

        if self.downsample:

            # Conv for Spatial Downsampling, i.e., stride = 2
            self.conv = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride = 2, padding=1)

            # Downsampling here is better than downsampling at bottleneck,
            # Reason : Information is ignored if done at bottleneck, here the "ignored" info actually is incorporated in some cell.

            # Conv for increasing feature maps in "X" when adding the residual to the "out"
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2)
        else:

            # else default Convolution
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding=1)


        self.out_bottleneck = nn.Conv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=1)

        #activation
        self.relu = nn.ReLU()

    def forward(self, X):

        #decreased featuremap representaiton
        out = self.in_bottleneck(X)
        out = self.relu(out)

        # Conventional CNN
        out = self.conv(out)
        out = self.relu(out)

        #increased featuremap representation
        out = self.out_bottleneck(out)

        # Residual Connection
        if self.downsample:
            # We increase feature maps and reduce spatial size to match dimensions
            X = self.downsample_conv(X)


        out = X + out

        out = self.relu(X)

        return out

class ResNet(nn.Module):

    def __init__(self, in_channels):
        '''
        config format:
            {feature_maps : count, ...}
        '''

        super().__init__()


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels = 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.blocks = nn.Sequential()

        self.blocks.append(ResidualBlock(64, 64, 1))
        self.blocks.append(ResidualBlock(64, 64, 1))
        self.blocks.append(ResidualBlock(64, 64, 1))
        self.blocks.append(ResidualBlock(64, 64, 1))
        self.blocks.append(ResidualBlock(64, 64, 1))
        self.blocks.append(ResidualBlock(64, 64, 1))

        self.blocks.append(ResidualBlock(64, 128, 2))
        self.blocks.append(ResidualBlock(128, 128, 1))
        self.blocks.append(ResidualBlock(128, 128, 1))
        self.blocks.append(ResidualBlock(128, 128, 1))
        self.blocks.append(ResidualBlock(128, 128, 1))
        self.blocks.append(ResidualBlock(128, 128, 1))
        self.blocks.append(ResidualBlock(128, 128, 1))
        self.blocks.append(ResidualBlock(128, 128, 1))

        self.blocks.append(ResidualBlock(128, 256, 2))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))
        self.blocks.append(ResidualBlock(256, 256, 1))

        self.blocks.append(ResidualBlock(256, 512, 2))
        self.blocks.append(ResidualBlock(512, 512, 1))
        self.blocks.append(ResidualBlock(512, 512, 1))
        self.blocks.append(ResidualBlock(512, 512, 1))
        self.blocks.append(ResidualBlock(512, 512, 1))
        self.blocks.append(ResidualBlock(512, 512, 1))

    def forward(self, X):
        out = self.conv1(X)
        out = self.pool(out)

        out = self.blocks(out)

        return out

class DeepResNet(nn.Module):
    def __init__(self, in_channels):
        '''
        config format:
            {feature_maps : count, ...}
        '''

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.blocks = nn.Sequential()

        self.blocks.append(BottleneckResidualBlock(64, 64, 256, 4))
        self.blocks.append(BottleneckResidualBlock(256, 64, 256, 1))
        self.blocks.append(BottleneckResidualBlock(256, 64, 256, 1))

        self.blocks.append(BottleneckResidualBlock(256, 128, 512, 2))
        self.blocks.append(BottleneckResidualBlock(512, 128, 512, 1))
        self.blocks.append(BottleneckResidualBlock(512, 128, 512, 1))
        self.blocks.append(BottleneckResidualBlock(512, 128, 512, 1))

        self.blocks.append(BottleneckResidualBlock(512, 256, 1024, 2))
        self.blocks.append(BottleneckResidualBlock(1024, 256, 1024, 1))
        self.blocks.append(BottleneckResidualBlock(1024, 256, 1024, 1))
        self.blocks.append(BottleneckResidualBlock(1024, 256, 1024, 1))
        self.blocks.append(BottleneckResidualBlock(1024, 256, 1024, 1))
        self.blocks.append(BottleneckResidualBlock(1024, 256, 1024, 1))

        self.blocks.append(BottleneckResidualBlock(1024, 512, 2048, 2))
        self.blocks.append(BottleneckResidualBlock(2048, 512, 2048, 1))
        self.blocks.append(BottleneckResidualBlock(2048, 512, 2048, 1))
        

        

    def forward(self, X):
        out = self.conv1(X)
        out = self.pool(out)

        out = self.blocks(out)

        return out


        

