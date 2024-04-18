import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class Bottleneck(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            downsampling: Union[nn.Sequential, None]=None, 
            stride: int = 1
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)

        self.downsampling = downsampling
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsampling is not None:
            identity = self.downsampling(identity)
        
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(
            self, 
            Block: Bottleneck, 
            block_list: list, 
            num_classes: int,
        ):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(Block, num_blocks=block_list[0], out_channels=64)
        self.layer2 = self._make_layer(Block, num_blocks=block_list[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(Block, num_blocks=block_list[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(Block, num_blocks=block_list[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    def _make_layer(
            self,
            Block: Bottleneck, 
            num_blocks: int,
            out_channels: int,
            stride: int = 1
        ):
        downsampling = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
            downsampling = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )

        layers.append(Block(self.in_channels, out_channels, downsampling, stride))
        self.in_channels = out_channels*4

        for _ in range(1, num_blocks):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
