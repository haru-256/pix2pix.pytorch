"""
Loss に関するクラスの定義
"""
import torch
import torch.nn as nn


class GANLoss:
    def __init__(self, gan_mode, device, real_label=1.0, fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan.
            device (torch.device) : デバイスオブジェクト
            real_label (bool) - - label for a real image
            fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.real_label = torch.tensor(real_label).to(device)
        self.fake_label = torch.tensor(fake_label).to(device)
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
            print("Use LSGAN")
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
            print("Use vanilla GAN")
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, pred, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            pred (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(pred)

    def __call__(self, pred, target_is_real, for_dis=True):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            pred (tensor) - - tpyically the prediction output from a discriminator.
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            for_dis (bool): discriminatorのためかどうか．hinge lossの時のみ効果がある

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss = self.loss(pred, target_tensor)

        assert loss.dim() == 0, "GANLossがスカラーでない．Got {}".format(loss.shape)
        if torch.isnan(loss):
            raise ValueError("Loss がNanです．Got {}".format(loss))

        return loss


class L1Loss:
    def __init__(self):
        """L1ロスを返すクラス．

        Args:
            lambnda_ (float): L1Loss に対する重み

        Returns:
            loss (tensor): L1Loss
        """
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()
        print("Use L1Loss")

    def __call__(self, fake_B, real_B):
        """
        L1Loss を計算する．
        Args:
            fake_B (torch.Tensor): 生成画像．shape : (N, C, H, W)
            real_B (torch.Tensor): 本物画像．shape : (N, C, H, W)

        Returns:
            Global Generator を実装
        """

        loss = self.loss(fake_B, real_B)
        assert loss.dim() == 0, "L1Lossがスカラーでない．Got {}".format(loss.shape)
        return loss
