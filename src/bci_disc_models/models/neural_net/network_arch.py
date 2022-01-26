from typing import Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torchvision.models import resnet18


def get_model(arch: str, n_classes: int, input_shape: Tuple[int]):
    logger.info(f"Get model: {arch=}, {n_classes=}, {input_shape=}")
    if arch == "simple-cnn-1d":
        return SimpleCNN1D(n_classes=n_classes, input_shape=input_shape)
    elif arch == "simple-cnn-2d":
        return SimpleCNN2D(n_classes=n_classes, input_shape=input_shape)
    elif arch == "resnet18":
        return ResNet18(n_classes=n_classes, input_shape=input_shape)
    elif arch == "eegnet":
        return EEGNet(n_classes=n_classes, input_shape=input_shape)
    elif arch == "convmixer":
        return ConvMixer(input_shape=input_shape, dim=768, depth=32, kernel_size=7, patch_size=7, n_classes=n_classes)
    else:
        raise ValueError()


# See https://github.com/locuslab/convmixer/blob/main/convmixer.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, input_shape, dim, depth, kernel_size=9, patch_size=7, n_classes=2):
        super().__init__()
        input_chans = input_shape[0]
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_chans, affine=False),
            nn.Conv1d(input_chans, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"), nn.GELU(), nn.BatchNorm1d(dim)
                        )
                    ),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm1d(dim),
                )
                for _ in range(depth)
            ],
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, n_classes),
            nn.LogSoftmax(-1),
        )

    def forward(self, data):
        return self.net(data)


class ResNet18(nn.Module):
    """ResNet18 expects data with shape (batch, channels, height, width).
    For now, begin with shape (batch, channels, time) and unsqueeze to (batch, 1, channels, time)
    """

    def __init__(self, n_classes: int, input_shape: Tuple[int], pretrained=True):
        super().__init__()
        self.net = resnet18(pretrained=pretrained, progress=False)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(512, n_classes)

    def forward(self, data):
        data = data.unsqueeze(1)
        return F.log_softmax(self.net(data), dim=-1)


def _CNN_N_DIM(n_dim, input_channels):
    if n_dim == 1:
        conv, bn = nn.Conv1d, nn.BatchNorm1d
    elif n_dim == 2:
        conv, bn = nn.Conv2d, nn.BatchNorm2d
    else:
        raise ValueError("Only 1D and 2D CNNs are supported")

    return nn.Sequential(
        conv(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
        bn(32),
        nn.SiLU(inplace=True),
        conv(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
        bn(64),
        nn.SiLU(inplace=True),
        conv(64, 128, kernel_size=3, stride=2, bias=False),
        bn(128),
        nn.SiLU(inplace=True),
        conv(128, 256, kernel_size=3, stride=2, bias=False),
        bn(256),
        nn.SiLU(inplace=True),
        conv(256, 512, kernel_size=3, stride=2, bias=False),
        nn.Flatten(),
    )


class _SimpleCNNXD(nn.Module):
    def __init__(self, n_dim: int, n_classes: int, input_shape: Tuple[int]):
        super().__init__()
        self.net = _CNN_N_DIM(n_dim=n_dim, input_channels=input_shape[0])
        _hidden_size = self.net(torch.zeros(1, *input_shape)).numel()
        self.classifier = nn.Sequential(
            nn.Linear(_hidden_size, n_classes),
            nn.LogSoftmax(-1),
        )

    def forward(self, data):
        return self.classifier(self.net(data))


class SimpleCNN2D(_SimpleCNNXD):
    """Expects data with shape (batch, channels, H, W).
    Suitable for EEG data in time-frequency domain."""

    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        input_shape = (1, *input_shape)
        super().__init__(n_dim=2, n_classes=n_classes, input_shape=input_shape)

    def forward(self, data):
        return super().forward(data.unsqueeze(1))


class SimpleCNN1D(_SimpleCNNXD):
    """Expects data with shape (batch, channels, time).
    Suitable for EEG data in time domain."""

    def __init__(self, n_classes: int, input_shape: Tuple[int]):
        super().__init__(n_dim=1, n_classes=n_classes, input_shape=input_shape)


class EEGNet(nn.Module):
    """
    Attempt to reproduce model from: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py#L55
    NOTE - Some differences from original EEGNet:
    - kernel_constraints. https://discuss.pytorch.org/t/kernel-constraint-similar-to-the-one-implemented-in-keras/49936
        (could use pytorch parametrizations, geotorch, or similar)
    - Separable Conv may not be exactly equivalent. Need to choose when to increase channels:
        F1 -> F1, F1 -> F2   VS   F1 -> F2, F2 -> F2
    - Fixed latent dimension size using a final conv
    """

    def __init__(
        self,
        n_classes: int,
        input_shape: Tuple[int],
        feature_dim: int = 64,
        dropout_rate=0.5,
        F1=8,
        F2=16,
    ):
        super().__init__()
        input_channels, input_time_length = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 16), padding=8, bias=False),  # Originally 64, padding=32
            nn.BatchNorm2d(F1),
            # NOTE - groups == in_chan corresponds to "Depthwise Conv"
            nn.Conv2d(F1, F1, (input_channels, 1), bias=False, groups=F1),
            nn.BatchNorm2d(F1),
            nn.SiLU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            # NOTE - two separate convolutions to implement one "Separable Conv"
            nn.Conv2d(F1, F2, (1, 8), bias=False, groups=1),  # Originally both 16 instead of 8
            nn.Conv2d(F2, F2, (8, 1), bias=False, groups=F2),
            nn.BatchNorm2d(F2),
            nn.SiLU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        shape_after_conv = self.conv(torch.zeros(1, 1, input_channels, input_time_length)).shape
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(F2, feature_dim, (shape_after_conv[2], shape_after_conv[3]), bias=True),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, n_classes),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add dummy channel dimension
        return self.classifier(self.feature_reduction(self.conv(x)))
