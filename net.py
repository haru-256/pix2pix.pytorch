import torch
import torch.nn as nn
from torch.nn import init


def weights_init(m, gain):
    """
    Initialize

    Parameres
    ------------------------------
    m: torch.nn.Module
        Module that means a layer.

    gain: float
        standard variation
    """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find('BatchNorm2d') != -1:
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L41-L62
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


class EncoderBlock(nn.Module):
    """
    convolution->batchnormalization->leaky_relu

    Parameters
    ---------------------------
    in_c: int
        the number of input channels
    out_c: int
        the number of output channels
    ks: int or tuple
        the size of kernel size
    stride: int
        the size of stride size
    n_pd: int
        the number of paddings
    norm_type: string or None
        if None represents not to apply Batch Norm. if `batch` represents to apply Batch Norm.
        if `instance` represents to apply Instance Norm

    """

    def __init__(self, in_c, out_c, ks=4, stride=2, n_pd=1, norm_type="instance"):
        super(EncoderBlock, self).__init__()

        if norm_type == "batch":
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c,
                          kernel_size=ks, stride=stride, padding=n_pd, bias=False),
                nn.BatchNorm2d(num_features=out_c),
                nn.LeakyReLU(negative_slope=0.2))
        elif norm_type == "instance":
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c,
                          kernel_size=ks, stride=stride, padding=n_pd, bias=False),
                nn.InstanceNorm2d(num_features=out_c,
                                  affine=False, track_running_stats=False),  # affine=False?
                nn.LeakyReLU(negative_slope=0.2))
        else:
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c,
                          kernel_size=ks, stride=stride, padding=n_pd),
                nn.LeakyReLU(negative_slope=0.2))

        self.block = block

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Encoder of U-Net
    """

    def __init__(self, ngf=64):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            EncoderBlock(in_c=3, out_c=ngf, norm_type=None),  # C64
            EncoderBlock(in_c=ngf, out_c=2*ngf),  # C128
            EncoderBlock(in_c=ngf*2, out_c=ngf*4),  # C256
            EncoderBlock(in_c=ngf*4, out_c=ngf*8),  # C512
            EncoderBlock(in_c=ngf*8, out_c=ngf*8),  # C512
            EncoderBlock(in_c=ngf*8, out_c=ngf*8),  # C512
            EncoderBlock(in_c=ngf*8, out_c=ngf*8),  # C512
            EncoderBlock(in_c=ngf*8, out_c=ngf*8)  # C512)
        )

    def forward(self, x):
        hs = [self.encoder[0](x)]
        for block in self.encoder[1:]:
            hs.append(block(hs[-1]))
            # print(hs[-1].shape)
        return hs


class DecoderBlock(nn.Module):
    """
    convolution->batchnormalization->dropout->relu

    Parameters
    ---------------------------
    in_c: int
        the number of input channels
    out_c: int
        the number of output channels
    ks: int or tuple
        the size of kernel size
    stride: int
        the size of stride size
    n_pd: int
        the number of paddings
    norm_type: string or None
        if None represents not to apply Batch Norm. if `batch` represents to apply Batch Norm.
        if `instance` represents to apply Instance Norm

    """

    def __init__(self, in_c, out_c, ks=4, stride=2, n_pd=1, norm_type='instance'):
        super(DecoderBlock, self).__init__()

        if norm_type == 'batch':
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                                   kernel_size=ks, stride=stride, padding=n_pd),
                nn.BatchNorm2d(num_features=out_c),
                nn.Dropout2d(p=0.5),
                nn.ReLU())
        elif norm_type == 'instance':
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                                   kernel_size=ks, stride=stride, padding=n_pd),
                nn.InstanceNorm2d(num_features=out_c,
                                  affine=False, track_running_stats=False),
                nn.Dropout2d(p=0.5),
                nn.ReLU())
        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                                   kernel_size=ks, stride=stride, padding=n_pd),
                nn.Dropout2d(p=0.5),
                nn.ReLU())

        self.block = block

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    """
    Decoder of U-Net
    """

    def __init__(self, ngf=64):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            DecoderBlock(in_c=ngf*8, out_c=ngf*8),  # CD512
            DecoderBlock(in_c=ngf*16, out_c=ngf*8),  # CD512
            DecoderBlock(in_c=ngf*16, out_c=ngf*8),  # CD512
            DecoderBlock(in_c=ngf*16, out_c=ngf*8),  # CD512
            DecoderBlock(in_c=ngf*16, out_c=ngf*4),  # CD256
            DecoderBlock(in_c=ngf*8, out_c=ngf*2),  # CD128
            DecoderBlock(in_c=ngf*4, out_c=ngf),  # CD64
            nn.ConvTranspose2d(in_channels=ngf*2, out_channels=3,
                               kernel_size=4, stride=2, padding=1),  # 論文ではConvolutionとしているがおそらく間違い
            nn.Tanh()
        )

    def forward(self, hs):
        """
        inference

        Parameter
        -------------------
        hs: list of torch.Tensor
            outputs of decoder to use for skip connect

        """
        # Decode
        hs_r = list(reversed(hs))
        h = self.decoder[0](hs_r[0])
        print(h.shape)
        for skip, block in zip(hs_r[1:], self.decoder[1:]):
            h = block(torch.cat((h, skip), dim=1))
            print(h.shape)
        return h


class UnetGenerator(nn.Module):
    """
    U-Net Generator

    Parameters
    ----------------------
    ngf: int
        the number of gen filters in first conv layer
    """

    def __init__(self, ngf=64):
        super(UnetGenerator, self).__init__()

        self.encoder = Encoder(ngf=ngf)
        self.decoder = Decoder(ngf=ngf)

    def forward(self, x):
        # Encode
        hs = self.encoder(x)
        # Decode
        output = self.decoder(hs)

        return output


if __name__ == "__main__":
    import pathlib
    from tensorboardX import SummaryWriter

    unet = UnetGenerator()
    path = pathlib.Path('graph')
    with SummaryWriter(path) as writer:
        dummy_input = torch.Tensor(10, 3, 256, 256)
        writer.add_graph(unet, dummy_input)
