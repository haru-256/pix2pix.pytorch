import torch
import torchsummary
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

    """

    def __init__(self, in_c, out_c, ks=4, stride=2, n_pd=1, isBN=True):
        super(EncoderBlock, self).__init__()

        if isBN:
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c,
                          kernel_size=ks, stride=stride, padding=n_pd, bias=False),
                # nn.InstanceNorm2d を使用すべき？ -> 論文ではBatchNorm でn=1やっていたのでしない
                nn.BatchNorm2d(num_features=out_c),
                nn.LeakyReLU(negative_slope=0.2))
        else:
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c,
                          kernel_size=ks, stride=stride, padding=n_pd),
                nn.LeakyReLU(negative_slope=0.2))

        self.block = block

    def forward(self, x):
        return self.block(x)


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

    """

    def __init__(self, in_c, out_c, ks=4, stride=2, n_pd=1, isBN=True):
        super(DecoderBlock, self).__init__()

        if isBN:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                                   kernel_size=ks, stride=stride, padding=n_pd),
                # nn.InstanceNorm2d を使用すべき？ -> 論文ではBatchNorm でn=1やっていたのでしない
                nn.BatchNorm2d(num_features=out_c),
                nn.Dropout2d(p=0.5),
                nn.ReLU())
        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                                   kernel_size=ks, stride=stride, padding=n_pd),
                nn.LeakyReLU())

        self.block = block

    def forward(self, x):
        return self.block(x)


class UnetGenerator(nn.Module):
    """
    U-Net Generator

    Parameters
    ----------------------
    ngf: int
        the number of gen filters in first conv layer
    """

    def __init__(self, ngf=64, ):
        super(UnetGenerator, self).__init__()

        # Encoder
        self.ec1 = EncoderBlock(in_c=3, out_c=ngf, isBN=False)  # C64
        self.ec2 = EncoderBlock(in_c=ngf, out_c=2*ngf)  # C128
        self.ec3 = EncoderBlock(in_c=ngf*2, out_c=ngf*4)  # C256
        self.ec4 = EncoderBlock(in_c=ngf*4, out_c=ngf*8)  # C512
        self.ec5 = EncoderBlock(in_c=ngf*8, out_c=ngf*8)  # C512
        self.ec6 = EncoderBlock(in_c=ngf*8, out_c=ngf*8)  # C512
        self.ec7 = EncoderBlock(in_c=ngf*8, out_c=ngf*8)  # C512
        self.ec8 = EncoderBlock(in_c=ngf*8, out_c=ngf*8)  # C512

        # Decoder
        self.dc1 = DecoderBlock(in_c=ngf*8, out_c=ngf*8)  # CD512
        self.dc2 = DecoderBlock(in_c=ngf*16, out_c=ngf*8)  # CD512
        self.dc3 = DecoderBlock(in_c=ngf*16, out_c=ngf*8)  # CD512
        self.dc4 = DecoderBlock(in_c=ngf*16, out_c=ngf*8)  # CD512
        self.dc5 = DecoderBlock(in_c=ngf*16, out_c=ngf*4)  # CD256
        self.dc6 = DecoderBlock(in_c=ngf*8, out_c=ngf*2)  # CD128
        self.dc7 = DecoderBlock(in_c=ngf*4, out_c=ngf)  # CD64
        self.dc8 = nn.ConvTranspose2d(in_channels=ngf*2, out_channels=3,
                                      kernel_size=3, stride=1, padding=1)  # 論文ではConvolutionとしているがおそらく間違い

    def forward(self, x):
        intermediates = []
        # Encode
        h = x
        for i, block in enumerate(self.children())
           if block.__class__.__name__ == 'DecoderBlock':
                break
            h = block(h)
            intermediates.append(h)
        h1 = self.ec1(x)
        h2 = self.ec2(h1)
        h3 = self.ec3(h2)
        h4 = self.ec4(h3)
        h5 = self.ec5(h4)
        h6 = self.ec6(h5)
        h7 = self.ec7(h6)
        h8 = self.ec8(h7)

        # Decode
        h9 = self.dc1(h8)
        h10 = self.dc2(torch.cat((h9, h7), dim=1))
        h11 = self.dc3(torch.cat((h10, h6), dim=1))
        h12 = self.dc4(torch.cat((h11, h5), dim=1))
        h13 = self.dc5(torch.cat((h12, h4), dim=1))
        h14 = self.dc6(torch.cat((h13, h3), dim=1))
        h15 = self.dc7(torch.cat((h14, h2), dim=1))
        h16 = self.dc8(torch.cat((h15, h1), dim=1))

        return torch.tanh(h16)
