"""
ネットワークに関するファイル

Generator, Discriminator に相当するクラスとそれを適するヘルパー関数
また，初期化や正規化手法を定義する関数がある．
"""
import functools
import torch
import torch.nn as nn
from torch.nn import init


def weights_init(m, gain=0.02):
    """Initialize parameres.

    Args:
        m (torch.nn.Module): Module that means a layer.
        gain (float): standard variation

    """
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, gain)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm2d") != -1:
        if hasattr(m, "weight"):
            init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, "bias"):
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("InstanceNorm2d") != -1 and m.weight is not None:
        if hasattr(m, "weight"):
            init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, "bias"):
            init.constant_(m.bias.data, 0.0)


def get_norm_layer(norm_type="batch", affine=True):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, option (learnable affine parameters). We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=affine, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=affine, track_running_stats=False
        )
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def define_G(
    input_nc,
    output_nc,
    ngf,
    device,
    num_downs=7,
    norm_type="instance",
    use_dropout=True,
    init_gain=0.02,
    affine=True,
    verbose=True,
):
    """Create a generator

    Args:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        device (torch.device) : ネットワークを転送するデバイスオブジェクト
        n_downG (int) : Generator のdownsampling の回数．通常pix2pixはボトルネック部分が1x1になるまで
            downsamplingします．よって，128x128の場合はn_down=7, 256x256の場合はn_down=8 が適切です．
        norm_type (str) -- the name of normalization layers used in the network: batch | instance
        use_dropout (bool) -- if use dropout layers.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        affine (bool)  -- 正規化手法（BatchNorm, InstanceNorm）に対してaffine引数に与えるもの
        spectral_norm (bool) -- spectral norm を適用するかどうか
        verbose (bool) -- Generator のネットワーク構造を出力するかどうか

    Returns
        a generator

    """
    norm_layer = get_norm_layer(norm_type=norm_type, affine=affine)

    netG = UnetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        num_downs=num_downs,
        ngf=ngf,
        norm_layer=norm_layer,
        use_dropout=use_dropout,
    )
    # 初期化&GPUへ転送
    netG.to(device)
    netG.apply(functools.partial(weights_init, gain=init_gain))

    # 出力
    if verbose:
        print("Generator".format, end="\n" + "=" * 50 + "\n")
        print(netG, end="\n\n")
    return netG


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=True,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_dropout : ドロップアウトを使用するかどうか．主に多様性を目的に使用される．

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # UnetGenerator の構築
        # もっとも内側のBlock を定義
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
        use_bias=False,
    ):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        """
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        """
        if input_nc is None:
            input_nc = outer_nc
        # if outermost:
        if outermost or innermost:  # 公式の実装ではbottleneck でbiasがFalseになっていた．
            downconv = nn.Conv2d(
                input_nc,
                inner_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,  # outermostのdownにはnoralize をかけない
            )
        else:
            downconv = nn.Conv2d(
                input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,  # normalize しないのでbiasをかける
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


def define_D(
    input_nc,
    ndf,
    device,
    n_layers=3,
    norm_type="instance",
    init_gain=0.02,
    affine=True,
    verbose=True,
):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        device (torch.device) -- デバイスオブジェクト
        n_layers (int)   -- the number of conv layers in the discriminator．70x70 PatchGAN はn_layers=3
        num_D (int)  -- Discriminatorの数
        norm_type (str)         -- the type of normalization layers used in the network.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        affine (bool)  -- 正規化手法（BatchNorm, InstanceNorm）に対してaffine引数に何を与えるか

    """
    norm_layer = get_norm_layer(norm_type=norm_type, affine=affine)
    netD = NLayerDiscriminator(
        input_nc=input_nc, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer
    )
    # 初期化&デバイスへ転送
    netD.to(device)
    netD.apply(functools.partial(weights_init, gain=init_gain))

    # 出力
    if verbose:
        print("Discriminator", end="\n" + "=" * 50 + "\n")
        print(netD, end="\n" + "=" * 50 + "\n\n")
    return netD


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_bias=False
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        """
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        """
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, A, B):
        """Standard forward."""
        input_ = torch.cat((A, B), dim=1)
        return self.model(input_)
