#!/usr/bin/python
import math
import torch.nn as nn

from typing import List
# Necessary import to be be able to instantiate resnet directly by importing this file.
from torchvision.models.resnet import (
    Bottleneck,
    BasicBlock
)


class ResNetEncoder(nn.Module):
    """
    Implement Resnet model encoder (no FC Layer)
    """

    def __init__(self, block: nn.Module, layers: List[int]) -> None:
        """
        Initialize the model

        :param block: the class used to instanciate a resnet block
        :param layers: the number of layers per block
        :return: the instanciated model.
        """
        self.inplanes = 64
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class MlpDecoder(nn.Module):
    """
    Implement basic 1 layer MLP decoder
    """

    def __init__(self, input_shape: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_shape, num_classes)

    def forward(self, x):
        # return class logits.
        x = self.fc(x)

        return x
