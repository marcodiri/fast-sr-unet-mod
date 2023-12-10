from functools import partial
from math import log2

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Dict, Iterable, List, Optional, Union
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from kornia.filters import filter2d
from torch import Tensor, nn

from attend import Attend
from models import BaseGenerator


# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def identity(t, *args, **kwargs):
    return t


def is_power_of_two(n):
    return log2(n).is_integer()


def null_iterator():
    while True:
        yield None


# activation functions
def leaky_relu(neg_slope=0.2):
    return nn.LeakyReLU(neg_slope)


# down and upsample
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


def Upsample(*args):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        Blur(),
    )


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


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, num_conv_kernels=0):
        super().__init__()
        self.proj = AdaptiveConv2DMod(
            dim, dim_out, kernel=3, num_conv_kernels=num_conv_kernels
        )
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, conv_mods_iter: Optional[Iterable] = None):
        conv_mods_iter = default(conv_mods_iter, null_iterator())

        x = self.proj(x, mod=next(conv_mods_iter), kernel_mod=next(conv_mods_iter))

        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self, dim, dim_out, *, groups=8, num_conv_kernels=0, style_dims: List = []
    ):
        super().__init__()
        style_dims.extend([dim, num_conv_kernels, dim_out, num_conv_kernels])

        self.block1 = Block(
            dim, dim_out, groups=groups, num_conv_kernels=num_conv_kernels
        )
        self.block2 = Block(
            dim_out, dim_out, groups=groups, num_conv_kernels=num_conv_kernels
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, conv_mods_iter: Optional[Iterable] = None):
        h = self.block1(x, conv_mods_iter=conv_mods_iter)
        h = self.block2(h, conv_mods_iter=conv_mods_iter)

        return h + self.res_conv(x)


# attention
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# feedforward
def FeedForward(dim, mult=4):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Conv2d(dim * mult, dim, 1),
    )


# transformers
class Transformer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, depth=1, flash_attn=True, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim, dim_head=dim_head, heads=heads, flash=flash_attn
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class LinearTransformer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, depth=1, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LinearAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


def get_same_padding(size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2


# style mapping network
class EqualLinear(nn.Module):
    def __init__(self, dim, dim_out, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleNetwork(nn.Module):
    def __init__(self, dim, depth, lr_mul=0.1, dim_text_latent=0):
        super().__init__()
        self.dim = dim
        self.dim_text_latent = dim_text_latent

        layers = []
        for i in range(depth):
            is_first = i == 0
            dim_in = (dim + dim_text_latent) if is_first else dim

            layers.extend([EqualLinear(dim_in, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x, text_latent=None):
        x = F.normalize(x, dim=1)

        if self.dim_text_latent > 0:
            assert exists(text_latent)
            x = torch.cat((x, text_latent), dim=-1)

        return self.net(x)


# adaptive conv
class AdaptiveConv2DMod(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod=True,
        stride=1,
        dilation=1,
        eps=1e-8,
        num_conv_kernels=1,  # set this to be greater than 1 for adaptive
    ):
        super().__init__()
        self.eps = eps

        self.dim_out = dim_out

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1

        self.weights = nn.Parameter(
            torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel))
        )

        self.demod = demod

        nn.init.kaiming_normal_(
            self.weights, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(
        self, fmap, mod: Optional[Tensor] = None, kernel_mod: Optional[Tensor] = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b, h = fmap.shape[0], fmap.shape[-2]

        # account for feature map that has been expanded by the scale in the first dimension
        # due to multiscale inputs and outputs

        if mod.shape[0] != b:
            mod = repeat(mod, "b ... -> (s b) ...", s=b // mod.shape[0])

        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0

            assert self.adaptive or not kernel_mod_has_el

            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(
                    kernel_mod, "b ... -> (s b) ...", s=b // kernel_mod.shape[0]
                )

        # prepare weights for modulation

        weights = self.weights

        if self.adaptive:
            weights = repeat(weights, "... -> b ...", b=b)

            # determine an adaptive weight and 'select' the kernel to use with softmax

            assert exists(kernel_mod) and kernel_mod.numel() > 0

            kernel_attn = kernel_mod.softmax(dim=-1)
            kernel_attn = rearrange(kernel_attn, "b n -> b n 1 1 1 1")

            weights = reduce(weights * kernel_attn, "b n ... -> b ...", "sum")

        # do the modulation, demodulation, as done in stylegan2

        mod = rearrange(mod, "b i -> b 1 i 1 1")

        weights = weights * (mod + 1)

        if self.demod:
            inv_norm = (
                reduce(weights**2, "b o i k1 k2 -> b o 1 1 1", "sum")
                .clamp(min=self.eps)
                .rsqrt()
            )
            weights = weights * inv_norm

        fmap = rearrange(fmap, "b c h w -> 1 (b c) h w")

        weights = rearrange(weights, "b o ... -> (b o) ...")

        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        fmap = F.conv2d(fmap, weights, padding=padding, groups=b)

        return rearrange(fmap, "1 (b o) ... -> b o ...", b=b)


# model
class UnetUpsampler(BaseGenerator):
    @beartype
    def __init__(
        self,
        dim,
        *,
        image_size,
        input_image_size,
        init_dim=None,
        out_dim=None,
        style_network: Optional[Union[StyleNetwork, Dict]] = None,
        style_network_dim=None,
        dim_mults=(1, 2, 4, 8, 16),
        channels=3,
        resnet_block_groups=8,
        full_attn=(False, False, False, True, True),
        flash_attn=True,
        self_attn_dim_head=64,
        self_attn_heads=8,
        self_attn_dot_product=True,
        self_attn_ff_mult=4,
        attn_depths=(1, 1, 1, 1, 1),
        cross_ff_mult=4,
        mid_attn_depth=1,
        num_conv_kernels=2,
        resize_mode="bilinear",
        unconditional=True,
        skip_connect_scale=None,
    ):
        super().__init__()

        # style network

        if isinstance(style_network, dict):
            style_network = StyleNetwork(**style_network)

        self.style_network = style_network

        assert exists(style_network) ^ exists(
            style_network_dim
        ), "either style_network or style_network_dim must be passed in"

        self.unconditional = unconditional

        assert is_power_of_two(image_size) and is_power_of_two(
            input_image_size
        ), "both output image size and input image size must be power of 2"
        assert (
            input_image_size < image_size
        ), "input image size must be smaller than the output image size, thus upsampling"

        num_layer_no_downsample = int(log2(image_size) - log2(input_image_size))
        assert num_layer_no_downsample <= len(
            dim_mults
        ), "you need more stages in this unet for the level of upsampling"

        self.image_size = image_size
        self.input_image_size = input_image_size

        # setup adaptive conv

        style_embed_split_dims = []

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        *_, mid_dim = dims

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            num_conv_kernels=num_conv_kernels,
            style_dims=style_embed_split_dims,
        )

        # attention

        full_attn = cast_tuple(full_attn, length=len(dim_mults))
        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Transformer, flash_attn=flash_attn)

        assert unconditional or len(full_attn) == len(dim_mults)

        # skip connection scale

        self.skip_connect_scale = default(skip_connect_scale, 2**-0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_depth) in enumerate(
            zip(in_out, full_attn, attn_depths)
        ):
            should_not_downsample = ind < num_layer_no_downsample

            attn_klass = FullAttention if layer_full_attn else LinearTransformer

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in),
                        block_klass(dim_in, dim_in),
                        attn_klass(
                            dim_in,
                            dim_head=self_attn_dim_head,
                            heads=self_attn_heads,
                            depth=layer_attn_depth,
                        ),
                        Downsample(dim_in, dim_out)
                        if not should_not_downsample
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = FullAttention(
            mid_dim,
            dim_head=self_attn_dim_head,
            heads=self_attn_heads,
            depth=mid_attn_depth,
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.mid_to_rgb = nn.Conv2d(mid_dim, channels, 1)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_depth) in enumerate(
            zip(reversed(in_out), reversed(full_attn), reversed(attn_depths))
        ):
            attn_klass = FullAttention if layer_full_attn else LinearTransformer

            self.ups.append(
                nn.ModuleList(
                    [
                        PixelShuffleUpsample(dim_out, dim_in),
                        Upsample(),
                        nn.Conv2d(dim_in, channels, 1),
                        block_klass(dim_in * 2, dim_in),
                        block_klass(dim_in * 2, dim_in),
                        attn_klass(
                            dim_in,
                            dim_head=self_attn_dim_head,
                            heads=self_attn_heads,
                            depth=layer_attn_depth,
                        ),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim, dim)

        self.final_to_rgb = nn.Conv2d(dim, channels, 1)

        # resize mode

        self.resize_mode = resize_mode

        # determine the projection of the style embedding to convolutional modulation weights (+ adaptive kernel selection weights) for all layers

        self.style_to_conv_modulations = nn.Linear(
            style_network.dim, sum(style_embed_split_dims)
        )
        self.style_embed_split_dims = style_embed_split_dims

    @property
    def allowable_rgb_resolutions(self):
        input_res_base = int(log2(self.input_image_size))
        output_res_base = int(log2(self.image_size))
        allowed_rgb_res_base = list(range(input_res_base, output_res_base))
        return [*map(lambda p: 2**p, allowed_rgb_res_base)]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    def resize_image_to(self, x, size):
        return F.interpolate(x, (size, size), mode=self.resize_mode)

    def forward(
        self,
        lowres_image,
        styles=None,
        noise=None,
        return_all_rgbs=False,
        replace_rgb_with_input_lowres_image=True,  # discriminator should also receive the low resolution image the upsampler sees
    ):
        x = lowres_image
        shape = x.shape
        batch_size = shape[0]

        assert shape[-2:] == ((self.input_image_size,) * 2)

        # styles

        if not exists(styles):
            assert exists(self.style_network)

            noise = default(
                noise,
                torch.randn((batch_size, self.style_network.dim), device=self.device),
            )
            styles = self.style_network(
                noise, None
            )  # TODO: try passing noise and lowres image

        # project styles to conv modulations

        conv_mods = self.style_to_conv_modulations(styles)
        conv_mods = conv_mods.split(self.style_embed_split_dims, dim=-1)
        conv_mods = iter(conv_mods)

        # initial conv

        x = self.init_conv(x)

        h = []

        # downsample stages

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, conv_mods_iter=conv_mods)
            h.append(x)

            x = block2(x, conv_mods_iter=conv_mods)

            x = attn(x)

            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, conv_mods_iter=conv_mods)
        x = self.mid_attn(x)
        x = self.mid_block2(x, conv_mods_iter=conv_mods)

        # rgbs

        rgbs = []

        init_rgb_shape = list(x.shape)
        init_rgb_shape[1] = self.channels

        rgb = self.mid_to_rgb(x)
        rgbs.append(rgb)

        # upsample stages

        for upsample, upsample_rgb, to_rgb, block1, block2, attn in self.ups:
            x = upsample(x)
            rgb = upsample_rgb(rgb)

            res1 = h.pop() * self.skip_connect_scale
            res2 = h.pop() * self.skip_connect_scale

            fmap_size = x.shape[-1]
            residual_fmap_size = res1.shape[-1]

            if residual_fmap_size != fmap_size:
                res1 = self.resize_image_to(res1, fmap_size)
                res2 = self.resize_image_to(res2, fmap_size)

            x = torch.cat((x, res1), dim=1)
            x = block1(x, conv_mods_iter=conv_mods)

            x = torch.cat((x, res2), dim=1)
            x = block2(x, conv_mods_iter=conv_mods)

            x = attn(x)

            rgb = rgb + to_rgb(x)
            rgbs.append(rgb)

        x = self.final_res_block(x, conv_mods_iter=conv_mods)

        assert len([*conv_mods]) == 0

        rgb = rgb + self.final_to_rgb(x)

        if not return_all_rgbs:
            return rgb

        # only keep those rgbs whose feature map is greater than the input image to be upsampled

        rgbs = list(filter(lambda t: t.shape[-1] > shape[-1], rgbs))

        # and return the original input image as the smallest rgb

        rgbs = [lowres_image, *rgbs]

        return rgb, rgbs
