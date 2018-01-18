#!/usr/bin/python
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class AlexNetEncoder(nn.Module):

    def __init__(self,
                 im_grayscale: bool=True,
                 im_gradients: bool=True) -> None:
        """
        Build AlexNet Architecture (Encoder part only). as in Noise as Target paper implementation.
        :param im_grayscale: whether to use grayscale or RGB input image.
        :param im_gradients: whether to use a fixed sobel operator at the start of the network instead of the raw pixels.
        """
        super().__init__()

        self.im_gradients = im_gradients
        n_input_channels = 1 if im_grayscale else 3
        self.n_input_channels = n_input_channels

        # Hard coded block that computes the image gradients from grayscale.
        conv_gradients = nn.Conv2d(n_input_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)

        if self.im_gradients:
            self.pre_processing = nn.Sequential(
                conv_gradients,
            )

        self.features = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.Conv2d(2, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self._initialise_weights()

        if self.im_gradients:
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
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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


class AlexNetDecoder(nn.Module):

    def __init__(self, input_shape: int, num_classes: int=10):
        """
        Build AlexNet Architecture (Decoder part only). as in Noise as Target paper implementation.
        """
        super().__init__()

        self.classifier = nn.Sequential(
                nn.Linear(input_shape, 4096),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                size = m.weight.size()
                n = size[0] + size[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.classifier(x)
