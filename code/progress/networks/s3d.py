from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation


class TemporalSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__(
            Conv3dNormActivation(
                in_planes,
                out_planes,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
                norm_layer=norm_layer,
            ),
            Conv3dNormActivation(
                out_planes,
                out_planes,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
                norm_layer=norm_layer,
            ),
        )


class SepInceptionBlock3D(nn.Module):
    def __init__(
        self,
        in_planes: int,
        b0_out: int,
        b1_mid: int,
        b1_out: int,
        b2_mid: int,
        b2_out: int,
        b3_out: int,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()

        self.branch0 = Conv3dNormActivation(
            in_planes, b0_out, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.branch1 = nn.Sequential(
            Conv3dNormActivation(
                in_planes, b1_mid, kernel_size=1, stride=1, norm_layer=norm_layer),
            TemporalSeparableConv(
                b1_mid, b1_out, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
        )
        self.branch2 = nn.Sequential(
            Conv3dNormActivation(
                in_planes, b2_mid, kernel_size=1, stride=1, norm_layer=norm_layer),
            TemporalSeparableConv(
                b2_mid, b2_out, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3dNormActivation(
                in_planes, b3_out, kernel_size=1, stride=1, norm_layer=norm_layer),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class S3D(nn.Module):
    """S3D main class.

    Args:
        num_class (int): number of classes for the classification task.
        dropout (float): dropout probability.
        norm_layer (Optional[Callable]): Module specifying the normalization layer to use.

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(
        self,
        num_classes: int = 400,
        dropout: float = 0.2,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm3d, eps=0.001, momentum=0.001)

        self.features = nn.Sequential(
            TemporalSeparableConv(3, 64, 7, 2, 3, norm_layer),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1)),
            Conv3dNormActivation(
                64,
                64,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
            ),
            TemporalSeparableConv(64, 192, 3, 1, 1, norm_layer),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1)),
            SepInceptionBlock3D(192, 64, 96, 128, 16, 32, 32, norm_layer),
            SepInceptionBlock3D(256, 128, 128, 192, 32, 96, 64, norm_layer),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(
                2, 2, 2), padding=(1, 1, 1)),
            SepInceptionBlock3D(480, 192, 96, 208, 16, 48, 64, norm_layer),
            # SepInceptionBlock3D(512, 160, 112, 224, 24, 64, 64, norm_layer),
            # SepInceptionBlock3D(512, 128, 128, 256, 24, 64, 64, norm_layer),
            # SepInceptionBlock3D(512, 112, 144, 288, 32, 64, 64, norm_layer),
            # SepInceptionBlock3D(528, 256, 160, 320, 32, 128, 128, norm_layer),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(
            #     2, 2, 2), padding=(0, 0, 0)),
            # SepInceptionBlock3D(832, 256, 160, 320, 32, 128, 128, norm_layer),
            # SepInceptionBlock3D(832, 384, 192, 384, 48, 128, 128, norm_layer),
        )
        self.avgpool = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            # nn.Linear(512, num_classes),
            nn.Conv3d(512, num_classes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        x = torch.mean(x, dim=(2, 3, 4))
        return x
