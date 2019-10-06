import argparse
import pathlib
import pickle
import torch


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.parser = argparse.ArgumentParser(
            prog="pose-to-image",
            usage="`python main.py` for training",
            description="train pix2pix with fashion550k",
            epilog="end",
            add_help=True,
        )
        self.parser = self.initialize(self.parser)

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("-s", "--seed", help="seed", type=int, required=True)
        parser.add_argument(
            "-n", "--num", help="the number of experiments.", type=int, required=True
        )
        parser.add_argument(
            "--dataroot",
            required=True,
            help="train_df.csv, val_df.csv, test_df.csv が格納されているディレクトリ．",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="edge2shoes",
            help="name of the experiment. It decides where to store samples and models．Default is edge2shoes",
        )
        parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,
            choices=[-1, 0, 1],
            help="gpu id: e.g. 0 , 1. use -1 for CPU. 一枚のGPUを使用する．" "複数枚には対応していない．",
        )
        # 入出力関係
        parser.add_argument(
            "--A_nc",
            type=int,
            default=3,
            help="# of input image channels: 3 for RGB and 1 for grayscale",
        )
        parser.add_argument("--nz", type=int, default=8, help="符号の次元8次元ベクトル")
        parser.add_argument(
            "--B_nc",
            type=int,
            default=3,
            help="# of output image channels: 3 for RGB and 1 for grayscale",
        )
        # model parameter
        # generator
        parser.add_argument(
            "--ngf",
            type=int,
            default=64,
            help="# of gen filters in the last conv layer. default is 64",
        )
        parser.add_argument(
            "--n_downG",
            type=int,
            default=7,
            help="Generator のダウンサンプリング回数．default = 7, for 128x128",
        )
        # Discriminator
        parser.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="# of discrim filters in the first conv layer",
        )
        parser.add_argument(
            "--n_layers_D",
            type=int,
            default=3,
            help="Discriminator の中間の層数．default is 3 , 70x70 PatchGAN",
        )
        parser.add_argument(
            "--norm_type",
            type=str,
            default="instance",
            help="instance normalization or batch normalization [instance | batch | none]",
        )
        parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="network initialization [normal]",
        )
        parser.add_argument(
            "--init_gain",
            type=float,
            default=0.02,
            help="scaling factor for normal initialization. default is 0.002",
        )
        parser.add_argument(
            "--no_dropout", action="store_true", help="no dropout for the generator"
        )
        parser.add_argument(
            "--no_affine",
            action="store_true",
            help="do not apply affine transformation.",
        )
        # dataset parameters
        parser.add_argument(
            "--nThreads",
            default=3,
            type=int,
            help="# threads for loading data. Default is 3",
        )
        parser.add_argument(
            "--batch_size", type=int, default=1, help="input batch size. default is 1"
        )
        parser.add_argument(
            "--scaleSize", type=int, default=256, help="画像の幅をこのサイズにscale．default is 128"
        )
        parser.add_argument(
            "--cropSize", type=int, default=256, help="crop images to this size"
        )
        parser.add_argument(
            "--preprocess",
            type=str,
            default="none",
            help="scaling and cropping of images at load time [scale_width | crop | scale_width_and_crop | none]. Default is scale_width",
        )
        parser.add_argument(
            "--no_flip",
            action="store_true",
            help="if specified, do not flip the images for data augmentation",
        )
        parser.add_argument(
            "-mA",
            "--meanA",
            help="mean to use for noarmalization of A",
            type=float,
            default=0,
        )
        parser.add_argument(
            "-stdA",
            "--stdA",
            help="std to use for noarmalization of A",
            type=float,
            default=1,
        )
        parser.add_argument(
            "-mB",
            "--meanB",
            help="mean to use for noarmalization of B",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "-stdB",
            "--stdB",
            help="std to use for noarmalization of B",
            type=float,
            default=0.5,
        )
        # additional parameters
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="if specified, print more debugging information",
        )
        parser.add_argument("--save_dir", help="重みや画像の保存先", required=True)
        self.initialized = True
        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        file_name = opt.checkpoints_dir / "args.txt"
        with open(file_name, "w") as f:
            f.write(message)
        file_name = opt.checkpoints_dir / "args.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(opt, f)

    def parse(self):
        """Parse our options, create checkpoints directory, and set up gpu device."""

        # parse arg
        opt = self.parser.parse_args()
        self.opt = opt

        # make checkpoint directory
        self.opt = opt
        checkpoints_dir = pathlib.Path(
            "{0}/{1}/result_{2}/result_{2}_{3}".format(
                opt.save_dir, opt.name, opt.num, opt.seed
            )
        ).resolve()  # 外付けハードディスクに設定
        for path in list(checkpoints_dir.parents)[::-1]:
            if not path.exists():
                path.mkdir()
        if not checkpoints_dir.exists():
            checkpoints_dir.mkdir()

        # make dir to save
        self.opt.model_dir = checkpoints_dir / "models"
        if not self.opt.model_dir.exists():
            self.opt.model_dir.mkdir()
        self.opt.log_dir = checkpoints_dir / "log"
        if not self.opt.log_dir.exists():
            self.opt.log_dir.mkdir()
        self.opt.image_dir = checkpoints_dir / "gen_images"
        if not self.opt.image_dir.exists():
            self.opt.image_dir.mkdir()

        print("init checkpoints_dir: {}".format(checkpoints_dir))
        self.opt.checkpoints_dir = checkpoints_dir

        # set gpu ids
        if opt.gpu_id == 0:
            device = torch.device("cuda:0")
        elif opt.gpu_id == 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        self.opt.device = device

        # オプションの出力&保存
        self.print_options(opt)

        return self.opt


if __name__ == "__main__":
    base_options = BaseOptions()
    opt = base_options.parse()
