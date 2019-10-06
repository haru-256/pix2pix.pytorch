import torch
from .networks import define_G, define_D
from .loss import GANLoss, L1Loss
from .base_model import BaseModel


class Pix2PixModel(BaseModel):
    def __init__(self, opt):
        """モデル，ロス，オプティマイザーに関する初期化

        Args:
            opt (Namespace): オプション
        """
        super(Pix2PixModel, self).__init__(opt)
        # モデルの定義
        # Generator
        self.netG = define_G(
            input_nc=opt.A_nc,
            output_nc=self.opt.output_nc,
            ngf=self.opt.ngf,
            device=self.device,
            num_downs=opt.n_downG,
            norm_type=self.opt.norm_type,
            use_dropout=not self.opt.no_dropout,
            init_gain=self.opt.init_gain,
            affine=not self.opt.no_affine,
        )

        # Discriminator
        self.netD = define_D(
            input_nc=opt.B_nc,
            ndf=self.opt.ndf,
            n_layers=self.opt.n_layersD,
            device=self.device,
            norm_type=self.opt.norm_type,
            init_gain=self.opt.init_gain,
            affine=not self.opt.no_affine,
        )

        # Optimizer の初期化
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )

        # resume であれば，pretrainのモデルを読み込む．
        self.opt.start_epoch = 1
        if self.opt.resume:
            if self.opt.which_epoch == "latest":
                path, latest_epoch = self.get_latest_file()
                self.opt.start_epoch = latest_epoch + 1
            else:
                self.opt.start_epoch = int(self.opt.which_epoch) + 1
                path = self.opt.model_dir / "pix2pix_{}epoch.tar".format(
                    self.opt.which_epoch
                )
                assert path.exists(), "Not Found : {}".format(path)
            checkpoint = torch.load(path, self.opt.device)

            self.netG.load_state_dict(checkpoint["gen_model_state_dict"])
            self.netD.load_state_dict(checkpoint["dis_model_state_dict"])
            self.optimizer_G.load_state_dict(checkpoint["gen_optim_state_dict"])
            self.optimizer_D.load_state_dict(checkpoint["dis_optim_state_dict"])

            print("Load pretrained model : {}".format(path), end="\n" + "=" * 60 + "\n")

        # ロス関数の初期化
        # GANLoss
        self.criterionGAN = GANLoss(gan_mode=self.opt.gan_mode, device=self.device)
        # L1Loss
        if not self.opt.no_l1loss:
            self.criterionL1 = L1Loss()

    def __call__(self, data_dict):
        """
        forward してloss を計算．

        Parameters
        ----------
        data_dict : dict of nn.Tensor
            学習データ．キーは "A"(入力), "B"（出力）.

        Returns
        -------
        losses : 各loss の Loss. Each keys is "g_gan", "g_l1", "d_real", "d_fake".
        """
        losses = {"g_gan": 0, "g_l1": 0, "d_real": 0, "d_fake": 0}

        # data migrate device
        data_dict = self.migrate(data_dict)
        # Generate fake image
        fake_B = self.netG(data_dict["A"])
        assert (
            (-1 <= fake_B) * (fake_B <= 1)
        ).all(), "input data to discriminator range is not from -1 to 1. Got: {}".format(
            (fake_B.min(), fake_B.max())
        )

        # Discrimintor Loss = GANLoss(fake) + GANLoss(real)
        pred_fake = self.netD(A=data_dict["A"], B=fake_B.detach())
        pred_real = self.netD(A=data_dict["A"], B=data_dict["B"])
        losses["d_fake"] = self.criterionGAN(pred_fake, target_is_real=False)
        losses["d_real"] = self.criterionGAN(pred_real, target_is_real=True)

        # Generator Loss = GANLoss(fake passability loss) + PeceptualLoss + FMLoss + L1Loss
        # GAN Loss
        pred_fake = self.netD(A=data_dict["A"], B=fake_B)
        losses["g_gan"] = self.criterionGAN(pred_fake, target_is_real=True)

        # L1 Loss
        if not self.opt.no_l1loss:
            losses["g_l1"] = self.criterionL1(fake_B, data_dict["A"])

        return losses
