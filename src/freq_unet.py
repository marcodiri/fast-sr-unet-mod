import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import functional as F

from models import BaseGenerator


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)

        conv = nn.Conv2d(dim, dim_out * 4, 1)
        self.init_conv_(conv)

        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))

    def init_conv_(self, conv):
        o, *rest_shape = conv.weight.shape
        conv_weight = torch.empty(o // 4, *rest_shape)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class contrast(nn.Module):
    def __init__(self, nIn, nOut, stride=1, d=1):
        super(contrast, self).__init__()
        self.d1 = nn.Conv2d(
            nIn,
            nOut,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=nOut,
            bias=True,
            dilation=1,
        )
        self.d2 = nn.Conv2d(
            nIn, nOut, kernel_size=3, stride=1, padding=6, bias=True, dilation=6
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y1 = self.lrelu(self.d1(x))
        y2 = self.lrelu(self.d2(x))
        df = torch.abs(y1 - y2)
        return self.act(df)


class contrast1(nn.Module):
    def __init__(self, nIn, nOut, stride=1, d=1):
        super(contrast1, self).__init__()
        self.d1 = nn.Conv2d(
            nIn,
            nOut,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=nOut,
            bias=True,
            dilation=1,
        )
        self.d2 = nn.Conv2d(
            nIn, nOut, kernel_size=3, stride=1, padding=12, bias=True, dilation=12
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y1 = self.lrelu(self.d1(x))
        y2 = self.lrelu(self.d2(x))
        df = torch.abs(y1 - y2)
        return self.act(df)


class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class AFDSRUnet(BaseGenerator):
    def __init__(
        self,
        in_dim=3,
        n_class=3,
        downsample=None,
        residual=False,
        batchnorm=False,
        scale_factor=2,
        n_filters=64,
        layer_multiplier=1,
    ):
        """
        Args:
            in_dim (float, optional):
                channel dimension of the input
            n_class (str):
                channel dimension of the output
            n_filters (int, optional):
                maximum number of filters. the layers start with n_filters / 2,  after each layer this number gets multiplied by 2
                during the encoding stage and until it reaches n_filters. During the decoding stage the number follows the reverse
                scheme. Default is 64
            downsample (None or float, optional)
                can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
            residual (bool):
                if using the residual scheme and adding the input to the final output
            scale_factor (int):
                upscale factor. if you want a rational upscale (e.g. 720p to 1080p, which is 1.5) combine it
                with the downsample parameter
            layer_multiplier (int or float):
                compress or extend the network depth in terms of total layers. configured as a multiplier to the number of the
                basic blocks which composes the layers
            batchnorm (bool, default=False):
                whether use batchnorm or not. If True should decrease quality and performances.
        """

        super().__init__()

        self.residual = residual
        self.n_class = n_class
        self.scale_factor = scale_factor

        self.conv1_1 = nn.Conv2d(in_dim, 32, 3, padding=1)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32) if batchnorm else nn.Identity()

        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32) if batchnorm else nn.Identity()

        self.conv2_1 = nn.Conv2d(35, 64, 3, padding=1)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64) if batchnorm else nn.Identity()
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64) if batchnorm else nn.Identity()

        self.conv3_1 = nn.Conv2d(67, 128, 3, padding=1)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128) if batchnorm else nn.Identity()
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128) if batchnorm else nn.Identity()

        self.conv4_1 = nn.Conv2d(131, 256, 3, padding=1)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256) if batchnorm else nn.Identity()
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256) if batchnorm else nn.Identity()

        self.conv5_1 = nn.Conv2d(259, 512, 3, padding=1)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = nn.BatchNorm2d(512) if batchnorm else nn.Identity()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = nn.BatchNorm2d(512) if batchnorm else nn.Identity()

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6_1 = nn.Conv2d(768, 256, 3, padding=1)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_1 = nn.BatchNorm2d(256) if batchnorm else nn.Identity()
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2 = nn.BatchNorm2d(256) if batchnorm else nn.Identity()

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_1 = nn.Conv2d(384, 128, 3, padding=1)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128) if batchnorm else nn.Identity()
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128) if batchnorm else nn.Identity()

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8_1 = nn.Conv2d(192, 64, 3, padding=1)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64) if batchnorm else nn.Identity()
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64) if batchnorm else nn.Identity()

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_1 = nn.Conv2d(96, 32, 3, padding=1)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32) if batchnorm else nn.Identity()
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.downsample = nn.MaxPool2d(2)
        self.upsample1 = PixelShuffleUpsample(32)
        self.upsample2 = PixelShuffleUpsample(32, n_class)

        self.ratio = nn.Parameter(torch.tensor(1.0), True)
        self.ratio1 = nn.Parameter(torch.tensor(1.0), True)
        self.ratio2 = nn.Parameter(torch.tensor(1.0), True)
        self.ratio3 = nn.Parameter(torch.tensor(1.0), True)
        self.ratio4 = nn.Parameter(torch.tensor(1.0), True)
        self.ratio5 = nn.Parameter(torch.tensor(1.0), True)
        self.lc5 = contrast(512, 512)
        self.lc4 = contrast(256, 256)
        self.lc3 = contrast(128, 128)
        self.lc2 = contrast(64, 64)
        self.lc1 = contrast(32, 32)
        self.lc51 = contrast1(512, 512)
        self.lc41 = contrast1(256, 256)
        self.lc31 = contrast1(128, 128)
        self.lc21 = contrast1(64, 64)
        self.lc11 = contrast1(32, 32)

        self.se5 = SEWeightModule(768)
        self.se4 = SEWeightModule(384)
        self.se3 = SEWeightModule(192)
        self.se2 = SEWeightModule(96)
        self.se1 = SEWeightModule(32)

    def forward(self, input):
        x = input

        x_down2 = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        x_down4 = F.interpolate(x_down2, scale_factor=0.5, mode="bilinear")
        x_down8 = F.interpolate(x_down4, scale_factor=0.5, mode="bilinear")
        x_down16 = F.interpolate(x_down8, scale_factor=0.5, mode="bilinear")

        x_reup8 = F.interpolate(x_down16, scale_factor=2, mode="bilinear")
        x_reup4 = F.interpolate(x_reup8, scale_factor=2, mode="bilinear")
        x_reup2 = F.interpolate(x_reup4, scale_factor=2, mode="bilinear")
        x_reup = F.interpolate(x_reup2, scale_factor=2, mode="bilinear")

        Laplace_8 = x_down8 - x_reup8
        Laplace_4 = x_down4 - x_reup4
        Laplace_2 = x_down2 - x_reup2
        Laplace_1 = x - x_reup

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(x)))

        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.downsample(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(torch.cat((x, Laplace_2), 1))))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.downsample(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(torch.cat((x, Laplace_4), 1))))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.downsample(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(torch.cat((x, Laplace_8), 1))))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        x = self.downsample(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(torch.cat((x, x_down16), 1))))
        x = (self.ratio - self.lc5(x)) * x

        conv1_0 = self.lc1(conv1) * conv1
        conv1_1 = self.lc11(conv1) * conv1
        conv2_0 = self.lc2(conv2) * conv2
        conv2_1 = self.lc21(conv2) * conv2
        conv3_0 = (self.ratio1 - self.lc3(conv3)) * conv3
        conv3_1 = (self.ratio2 - self.lc31(conv3)) * conv3
        conv4_0 = (self.ratio3 - self.lc4(conv4)) * conv4
        conv4_1 = (self.ratio4 - self.lc41(conv4)) * conv4

        conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
        conv5 = F.interpolate(conv5, scale_factor=2, mode="bilinear")

        up6 = torch.cat([self.deconv5(conv5), conv4_0, conv4_1], 1)
        up6se = self.se5(up6)
        up6 = up6se * up6
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

        conv6 = F.interpolate(conv6, scale_factor=2, mode="bilinear")
        up7 = torch.cat([self.deconv6(conv6), conv3_0, conv3_1], 1)
        up7se = self.se4(up7)
        up7 = up7se * up7
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

        conv7 = F.interpolate(conv7, scale_factor=2, mode="bilinear")
        up8 = torch.cat([self.deconv7(conv7), conv2_0, conv2_1], 1)
        up8se = self.se3(up8)
        up8 = up8se * up8
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

        conv8 = F.interpolate(conv8, scale_factor=2, mode="bilinear")
        up9 = torch.cat([self.deconv8(conv8), conv1_0, conv1_1], 1)
        up9se = self.se2(up9)
        up9 = up9se * up9
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        x = self.LReLU9_2(self.conv9_2(x))

        sf = self.scale_factor
        if sf == 4:
            x = self.upsample1(x)

        x = self.upsample2(x)

        if self.residual:
            x += self.ratio * F.interpolate(
                input[:, -self.n_class :, :, :], scale_factor=sf, mode="bicubic"
            )

        return F.tanh(x)
