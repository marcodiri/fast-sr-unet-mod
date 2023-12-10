# courtesy of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

import math

import lightning as L
import lpips  # courtesy of https://github.com/richzhang/PerceptualSimilarity
import numpy as np
import torch
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

import pytorch_ssim  # courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, input):
        out = self.conv1(input)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        return out


class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        batch_norm=False,
        activation=None,
        dilation=1,
        groups=1,
        use_spectral_norm=False,
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()
        # if groups is None:
        #    groups = 1
        if activation is not None:
            activation = activation.lower()
            assert activation in {"prelu", "leakyrelu", "tanh"}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        if not use_spectral_norm:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=groups,
                )
            )
        else:
            layers.append(
                spectral_norm(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2 + dilation // 2,
                        groups=groups,
                        dilation=1,
                    )
                )
            )

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An activation layer, if wanted
        if activation == "prelu":
            layers.append(nn.PReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2))
        elif activation == "tanh":
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * (scaling_factor**2),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(
            output
        )  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(
            output
        )  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation="leakyrelu",
        )

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation=None,
        )

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SRResNet(nn.Module):
    """
    The SRResNet, as defined in the paper.
    """

    def __init__(
        self,
        in_channels=3,
        large_kernel_size=9,
        small_kernel_size=3,
        n_channels=64,
        n_blocks=16,
        scaling_factor=2,
    ):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super(SRResNet, self).__init__()

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {1, 2, 4, 8}, "The scaling factor must be 2, 4, or 8!"
        self.scale_factor = scaling_factor
        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="leakyrelu",
            stride=1,
        )

        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels)
                for i in range(n_blocks)
            ]
        )

        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=False,
            activation="leakyrelu",
        )

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        # self.last_part = nn.Sequential(
        #    nn.Conv2d(n_channels, 3 * (scaling_factor ** 2), kernel_size=3, padding=3 // 2),
        #    nn.PixelShuffle(scaling_factor) if scaling_factor > 1 else nn.Identity(),
        # nn.Tanh()
        # )

        self.last_part = nn.Sequential(
            *[
                SubPixelConvolutionalBlock(
                    kernel_size=small_kernel_size,
                    n_channels=n_channels,
                    scaling_factor=2,
                )
                for i in range(n_subpixel_convolution_blocks)
            ]
        )

        # The last convolutional block
        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="Tanh",
        )

    def forward(self, lr_imgs):
        """
        Forward prop.

        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
        residual = output  # (N, n_channels, w, h)
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)
        self.hidden = output
        sr_imgs = self.last_part(
            output
        )  # (N, n_channels, w * scaling factor, h * scaling factor)
        # sr_imgs = sr_imgs + F.interpolate(lr_imgs,
        #                                 scale_factor=self.scale_factor,
        #                                 mode='bilinear')

        # sr_imgs = torch.clamp(sr_imgs, min=-1, max=1)
        # sr_imgs = torch.clamp(sr_imgs, min=-1, max=1)
        # self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
        #                                      batch_norm=False, activation='Tanh')

        return self.conv_block3(sr_imgs)


def cat_tensor(t1, t2):
    return torch.cat([t1, t2], dim=1)


class BaseGenerator(nn.Module):
    pass


class Generator(BaseGenerator):
    """
    The generator in the SRGAN, as defined in the paper. Architecture identical to the SRResNet.
    """

    def __init__(
        self,
        in_channels=3,
        large_kernel_size=7,
        small_kernel_size=3,
        n_channels=64,
        n_blocks=8,
        scaling_factor=2,
        downsample=None,
    ):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super(Generator, self).__init__()

        # The generator is simply an SRResNet, as above
        self.net = SRResNet(
            in_channels=in_channels,
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )

        if downsample is not None and downsample != scaling_factor:
            self.downsample = nn.Upsample(
                scale_factor=downsample, mode="bicubic", align_corners=True
            )
        else:
            self.downsample = nn.Identity()

    def initialize_with_srresnet(self, srresnet_checkpoint):
        """
        Initialize with weights from a trained SRResNet.

        :param srresnet_checkpoint: checkpoint filepath
        """
        srresnet = torch.load(srresnet_checkpoint)["model"]
        self.net.load_state_dict(srresnet.state_dict())

        print("\nLoaded weights from pre-trained SRResNet.\n")

    def forward(self, lr_imgs):
        """
        Forward prop.

        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(
            lr_imgs
        )  # (N, n_channels, w * scaling factor, h * scaling factor)

        return self.downsample(sr_imgs)


class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self, kernel_size=3, n_channels=32, n_blocks=8, fc_size=1024):
        """
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(Discriminator, self).__init__()

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (
                (n_channels if i is 0 else in_channels * 2)
                if i % 2 is 0
                else in_channels
            )
            conv_blocks.append(
                ConvolutionalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 is 0 else 2,
                    batch_norm=True,
                    activation="LeakyReLu",
                    use_spectral_norm=False,
                )
            )
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(fc_size, 1)

        # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

    def forward(self, imgs):
        """
        Forward propagation.

        :param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class GANModule(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: Discriminator,
        lpips_loss_weight=1.0,
        ssim_loss_weight=1.0,
        bce_loss_weight=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.G = generator
        self.D = discriminator

        self.lpips_loss = lpips.LPIPS(net="vgg", version="0.1")
        self.lpips_alex = lpips.LPIPS(net="alex", version="0.1")
        self.ssim = pytorch_ssim.SSIM()

        self.ssim_validation = []
        self.lpips_validation = []

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        g_opt, d_opt = self.optimizers()

        x, y_true = batch

        # train discriminator phase
        d_opt.zero_grad()

        y_fake = self.G(x)

        # train critic phase
        d_opt.zero_grad()

        batch_dim = x.shape[0]

        pred_true = self.D(y_true)

        # forward pass on true
        loss_true = F.binary_cross_entropy_with_logits(
            pred_true, torch.ones_like(pred_true, device=self.device)
        )

        # then updates on fakes
        pred_fake = self.D(y_fake.detach())
        loss_fake = F.binary_cross_entropy_with_logits(
            pred_fake, torch.zeros_like(pred_fake, device=self.device)
        )

        loss_discr = loss_true + loss_fake
        loss_discr *= 0.5
        self.manual_backward(loss_discr)
        d_opt.step()

        ## train generator phase
        g_opt.zero_grad()

        lpips_loss_ = self.lpips_loss(y_fake, y_true).mean()
        ssim_loss = 1.0 - self.ssim(y_fake, y_true)
        pred_fake = self.D(y_fake)
        bce = F.binary_cross_entropy_with_logits(
            pred_fake, torch.ones_like(pred_fake, device=self.device)
        )
        content_loss = (
            self.hparams.lpips_loss_weight * lpips_loss_
            + self.hparams.ssim_loss_weight * ssim_loss
        )
        loss_gen = content_loss + self.hparams.bce_loss_weight * bce

        self.manual_backward(loss_gen)
        g_opt.step()

        self.log_dict(
            {
                "g_loss": loss_gen,
                "d_loss": loss_discr,
                "content_loss": content_loss,
                "bce_loss": bce,
            },
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y_true = batch

        y_fake = self.G(x)

        ssim_val = self.ssim(y_fake, y_true).mean()
        lpips_val = self.lpips_alex(y_fake, y_true).mean()
        self.ssim_validation.append(float(ssim_val))
        self.lpips_validation.append(float(lpips_val))

        return x, y_true, y_fake

    def on_validation_epoch_end(self) -> None:
        ssim_mean = np.array(self.ssim_validation).mean()
        lpips_mean = np.array(self.lpips_validation).mean()

        self.log_dict(
            {
                "val_ssim": ssim_mean,
                "val_lpips": lpips_mean,
            },
            prog_bar=True,
        )

        self.ssim_validation.clear()
        self.lpips_validation.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        g_opt = torch.optim.Adam(params=self.G.parameters(), lr=1e-4)
        d_opt = torch.optim.Adam(params=self.D.parameters(), lr=1e-4)

        return g_opt, d_opt


class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.

    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """

    def __init__(self, i=5, j=4):
        """
        :param i: the index i in the definition above
        :param j: the index j in the definition above
        """
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG19 available in torchvision
        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert (
            maxpool_counter == i - 1 and conv_counter == j
        ), "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(
            *list(vgg19.features.children())[: truncate_at + 1]
        )

    def forward(self, input):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(
            input
        )  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
