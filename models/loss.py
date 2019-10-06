"""
Loss に関するクラスの定義
"""
import torch
import torch.nn as nn


class GANLoss:
    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            device (torch.device) : デバイスオブジェクト
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label).to(device))
        self.register_buffer("fake_label", torch.tensor(target_fake_label).to(device))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
            print("Use LSGAN")
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
            print("Use vanilla GAN")
        elif gan_mode == "hinge":
            print("Use Hinge GAN")
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, input_, target_is_real, for_dis=True):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor or list of tensor) - - tpyically the prediction output from a discriminator.
                list の場合，prediction[-1]がDiscriminatorの出力になっている必要がある．
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            for_dis (bool): discriminatorのためかどうか．hinge lossの時のみ効果がある

        Returns:
            the calculated loss.
        """
        if self.gan_mode == "hinge":
            if for_dis:
                # multiscale discriminator.
                loss = 0
                for input_i in input_:
                    pred = input_i[-1]
                    assert pred.size(1) == 1, "Discriminatorの最終出力のチャンネルが１でない"
                    if target_is_real:
                        loss += torch.mean(torch.relu(-(pred - 1)))
                    else:
                        loss += torch.mean(torch.relu(-(-pred - 1)))
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                # multiscale discriminator.
                loss = 0
                for input_i in input_:
                    pred = input_i[-1]
                    assert pred.size(1) == 1, "Discriminatorの最終出力のチャンネルが１でない"
                    loss += -torch.mean(pred)
        else:
            # multiscale discriminator.
            loss = 0
            for input_i in input_:
                pred = input_i[-1]
                assert pred.size(1) == 1, "Discriminatorの最終出力のチャンネルが１でない．Got {}".format(
                    pred.size(1)
                )
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)

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
