import torch.nn as nn
import torch.nn.functional as F
import torch


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)




class Bottleneck(nn.Module):
    """Standard Resnet3D Bottleneck.

    Args:
        in_planes (int): The number of input channels.
        planes (int): The number of output channels.
        stride (int | tuple(int)): The stride for the 3x3x3 convolution. Default: 1.
    """

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, int(in_planes/ self.expansion))
        self.bn1 = nn.BatchNorm3d(int(in_planes/ self.expansion))
        self.conv2 = conv3x3x3(int(in_planes/ self.expansion), int(in_planes/ self.expansion), stride)
        self.bn2 = nn.BatchNorm3d(int(in_planes/ self.expansion))
        self.conv3 = conv1x1x1(int(in_planes/ self.expansion), planes )
        self.bn3 = nn.BatchNorm3d(planes )
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Conv3d(in_planes, planes,1,1)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.in_planes != self.planes and self.stride==1:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DepthHead(nn.Module):
    """A U-net based implementation for the depth decoder that uses MMAction2's
    Resnet3D CSN implementation as its encoder.
    """

    def __init__(self):
        super(DepthHead, self,).__init__()

        self.decoder1 = nn.Sequential(
                        Bottleneck(2048,2048,1),
                        nn.ConvTranspose3d(2048,1024,(2,2,2),2)
                        )

        # 1024+1024 = 2048
        self.decoder2= nn.Sequential(
                        Bottleneck(2048,1024,1),
                        Bottleneck(1024,1024,1),
                        nn.ConvTranspose3d(1024,512,(2,2,2),2)
                        )

        # 512+512 = 1024
        self.decoder3 = nn.Sequential(
                        Bottleneck(1024,512,1),

                        Bottleneck(512,512,1),
                        nn.ConvTranspose3d(512,256,(2,2,2),2)
                        )

        # 256+256 = 512
        self.decoder4 = nn.Sequential(
                        Bottleneck(512,256,1),

                        nn.ConvTranspose3d(256,64,(3,2,2),(1,2,2), padding=(1,0,0))             
                        )     

        self.outc =  nn.ConvTranspose3d(64+64,1,(3,2,2),(1,2,2),(1,0,0))


    def forward(self,  x):
        # x0 is the encoder's first layer output and x4 is its last
        x0,x1,x2,x3,x4 = x 
        out = self.decoder1(x4)
        out = torch.concat((out,x3),dim=1)
        out = self.decoder2(out)
        out = torch.concat((out,x2),dim=1)
        out = self.decoder3(out)
        out = torch.concat((out,x1),dim=1)
        out = self.decoder4(out)
        out = torch.concat((out,x0),dim=1)
        out = self.outc(out)

        return out