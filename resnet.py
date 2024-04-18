from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsampling: Union[nn.Sequential, None] = None,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

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

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsampling: Union[nn.Sequential, None] = None,
        stride: int = 1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride,
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsampling = downsampling
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identify = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsampling is not None:
            identify = self.downsampling(identify)
        x += identify
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(
        self,
        Block: Union[type[Bottleneck], type[BasicBlock]],
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
        self.layer2 = self._make_layer(
            Block, num_blocks=block_list[1], out_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            Block, num_blocks=block_list[2], out_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            Block, num_blocks=block_list[3], out_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        self, Block: Union[type[Bottleneck], type[BasicBlock]], num_blocks: int, out_channels: int, stride: int = 1
    ):
        downsampling = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * Block.expansion:
            downsampling = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * Block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * Block.expansion),
            )

        layers.append(Block(self.in_channels, out_channels, downsampling, stride))
        self.in_channels = out_channels * Block.expansion

        for _ in range(1, num_blocks):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
