from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # トレーニングについて
        parser.add_argument("--epoch", type=int, default=30, help="epoch")
        parser.add_argument("--which_epoch", type=str, default="latest")
        # 最適化パラメーター
        parser.add_argument(
            "--beta1", type=float, default=0.5, help="momentum term of adam"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--gan_mode",
            type=str,
            default="vanilla",
            choices=["vanilla", "lsgan"],
            help="the type of GAN objective. [vanilla| lsgan]."
            " vanilla GAN loss is the cross-entropy objective used in the original GAN paper.",
        )
        # option of loss
        # Generator Loss
        parser.add_argument(
            "--no_l1loss", action="store_true", help="generatorのLossにl1Lossを定義しない"
        )
        parser.add_argument(
            "--lambda_l1", type=float, default=100.0, help="weight for l1 loss"
        )
        # Discriminator Loss
        parser.add_argument(
            "--coeff4dis",
            type=float,
            default=1.0,
            help="Discriminator のGANLoss にかけられる係数．公式リポジトリでは0.5がかけられてあった．default is 1.0",
        )
        # option for generated images
        parser.add_argument(
            "--vis_num", type=int, default=4 * 4, help="可視化する生成データの数, デフォルトは16"
        )
        parser.add_argument(
            "--save_freq",
            type=int,
            default=1,
            help="何epochごとにmodelや生成画像を保存するか．default is 1",
        )

        self.isTrain = True
        self.resume = False
        return parser


if __name__ == "__main__":
    from base_options import BaseOptions

    train_options = TrainOptions()
    opt = train_options.parse()
