#!/usr/bin/python
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
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

    def __init__(self,
                 block: nn.Module,
                 layers: List[int],
                 im_grayscale: bool=True,
                 im_gradients: bool=True) -> None:
        """
        Initialize the model

        :param block: the class used to instantiate a resnet block
        :param layers: the number of layers per block
        :param im_grayscale: whether the input image is grayscale or RGB.
        :param im_gradients: whether to use a fixed sobel operator at the start of the network instead of the raw pixels.
        """
        self.inplanes = 64
        super().__init__()

        self.im_gradients = im_gradients
        n_input_channels = 1 if im_grayscale else 3
        self.n_input_channels = n_input_channels

        # Hard coded block that computes the image gradients from grayscale.
        # Not learnt.
        conv_gradients = nn.Conv2d(n_input_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)

        if self.im_gradients:
            self.pre_processing = nn.Sequential(
                conv_gradients,
            )

        self.features = nn.Sequential(
            nn.Conv2d(2 if self.im_gradients else n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self._initialise_weights()

        if self.im_gradients:
            # override weights for preprocessing part.
            dx = np.array([[[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]]], dtype=np.float32)
            dy = np.array([[[-1.0, -2.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 1.0]]], dtype=np.float32)
            _conv_grad = torch.from_numpy(
                np.repeat(
                    np.concatenate([dx, dy])[:, np.newaxis, :, :],
                    n_input_channels,
                    axis=1
                )
            )
            conv_gradients.weight = nn.Parameter(data=_conv_grad, requires_grad=False)

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
        if self.im_gradients:
            x = self.pre_processing(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def get_output_shape(self, input_shape):
        assert len(input_shape) == 2, "Expects a 2 dimensional input shape."
        mock = np.random.random(tuple([2, self.n_input_channels]+list(input_shape))).astype(np.float32)
        v_mock = Variable(torch.from_numpy(mock))
        return self.forward(v_mock).size(1)


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
