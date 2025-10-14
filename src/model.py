import torch
import torch.nn as nn

class UNet(nn.Module):
    """
        Defines the UNet model class handling all of the architecture

        Implements the entire
    """
    def __init__(self, initial_channels, num_classes, features, batchnorm):
        super().__init__()

        self.initial_channels = initial_channels
        self.num_classes = num_classes
        self.features = features
        self.batchnorm = batchnorm

        # defines the down sample for each of the four downblocks
        self.down1 = DownBlock(self.initial_channels, self.features, True)
        self.down2 = DownBlock(self.features, self.features * 2, True)
        self.down3 = DownBlock(self.features * 2, self.features * 4, True)
        self.down4 = DownBlock(self.features * 4, self.features * 8, True)

        # defines the up sample for each of the four upblocks
        self.up1 = UpBlock(self.features * 16, self.features * 8, True)
        self.up2 = UpBlock(self.features * 8, self.features * 4, True)
        self.up3 = UpBlock(self.features * 4, self.features * 2, True)
        self.up4 = UpBlock(self.features * 2, self.features, True)

        # defines the bottleneck step
        self.bottleneck = DownBlock(self.features * 8, self.features * 16)

        # defines the final 1x1 convolution
        self.final_conv = nn.Conv2d(features, num_classes, kernel_size=1)


    def forward(self, x):
        """
            Defines the entire forward pass of the U-Net architecture

            Args:
                x: the input tensor
        """
        # define the pooling function (halves the feature size)
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # applies all four of the downsample forward passes
        skip1 = self.down1(x)
        x = pool(skip1)
        skip2 = self.down2(x)
        x = pool(skip2)
        skip3 = self.down3(x)
        x = pool(skip3)
        skip4 = self.down4(x)
        x = pool(skip4)

        # applies the bottleneck pass
        x = self.bottleneck(x)

        # applies the four upsample forward passses
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # applies the final 1x1 convolution
        x = self.final_conv(x)

        return x



class DownBlock(nn.Module):
    """
        A downsampling block used in the encoder of a U-Net

        Applies two convolutional layers followed by batch normalization and 
        ReLU activation. Padding and strides chosen to preserve feature size.

        Methods:
            forward(x):
                defines the forward pass of the block.
    """
    def __init__(self, in_channels, out_channels, batchnorm=True):
        """
            Defines initialization of a Downblock object

            Args:
                in_channels: number of incoming channels
                out_channels: number of outgoing channels
                batchnorm: boolean representing whether or not batch normalizaiton is applied
        """
        super(DownBlock, self).__init__()
        self.in_channels = in_channels # stores the number of channels for the convolutions
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1) # defines the convolution for the downblocks (padding to preserve feature size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        if (batchnorm): # defining batch normalization 
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    def forward(self, input_tensor):
        """
            Applies the convolutions and ReLU activations defined 
            in __init__.
            
            Args:
                input_tensor: the tensor representation of the input image
        """
        x = self.conv1(input_tensor)
        if (self.batchnorm):
            x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        if (self.batchnorm):
            x = self.bn2(x)
        x = nn.ReLU()(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        """
            Initialization of an upsampling block in the decoder

            Args:
                in_channels: number of incoming channels
                out_channels: number of outgoing channels
                batchnorm: boolean representation of whether batch normalization will be applied
            
            Methods:
                forward(x, y):
                    defines the forward pass of a single upscaling block in decoder
        """
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # defining upscaling convolution
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, stride=1, padding=1) # defining the regular convolution
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        if (batchnorm): # creates batch normalization objects for if batchnorm is true
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
            self.bn3 = nn.BatchNorm2d(num_features=out_channels)
    def forward(self, input_tensor, skip_tensor):
        """
            Applies the transposed convolution and two regular convolutions
            to the input_tensor, as well as the ReLU activation function and
            concatenations.

            Args:
                input_tensor: the tensor representation of the input image
                skip_tensor: the skip tensor from the downsampling step
        """
        x = self.conv1(input_tensor)
        if (self.batchnorm):
            x = self.bn1(x)
        x = nn.ReLU()(x)
        x = torch.cat([x, skip_tensor], 1)
        x = self.conv2(x)
        if (self.batchnorm):
            x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        if (self.batchnorm):
            x = self.bn3(x)
        x = nn.ReLU()(x)
        return x
