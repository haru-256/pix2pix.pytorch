from .base_options import BaseOptions
import argparse
import pickle
import pathlib
import torch


class ResumeOptions:
    """This class includes training options & resume options.

    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="再学習",
            usage="`python resume.py` for resume",
            description="resume training of pix2pix",
            epilog="end",
            add_help=True,
        )
        self.parser = self.initialize(self.parser)

    def initialize(self, parser):
        # resume を行うディレクトリに関して
        parser.add_argument("--save_dir", help="重みや画像の保存先のroot ディレクトリ", required=True)
        parser.add_argument(
            "--name",
            type=str,
            default="edges2shoes",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument("-s", "--seed", help="seed", type=int, required=True)
        parser.add_argument(
            "-n", "--num", help="the number of experiments.", type=int, required=True
        )
        parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,
            choices=[-1, 0, 1],
            help="gpu id: e.g. 0 , 1. use -1 for CPU. 一枚のGPUを使用する．" "複数枚には対応していない．",
        )
        parser.add_argument(
            "--epoch", type=int, default=50, help="あとepochまで行うか, default is 50"
        )
        parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="どのepochから始めるか．defaultはlatest",
        )

        self.isTrain = True
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
            # default = self.parser.get_default(k)
            # if v != default:
            #     comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        file_name = opt.checkpoints_dir / "args4resume.txt"
        with open(file_name, "w") as f:
            f.write(message)
        file_name = opt.checkpoints_dir / "args4resume.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(opt, f)

    def parse(self):
        """Parse our options, create checkpoints directory, and set up gpu device."""

        # parse arg
        opt = self.parser.parse_args()

        # checkpoint directory
        checkpoints_dir = pathlib.Path(
            "{0}/{1}/result_{2}/result_{2}_{3}".format(
                opt.save_dir, opt.name, opt.num, opt.seed
            )
        ).resolve()
        if not checkpoints_dir.exists():
            raise FileNotFoundError(
                "{} が存在せず，train optionが読み込めません".format(checkpoints_dir)
            )

        # optionを読み込み
        with open(checkpoints_dir / "args.pickle", "rb") as f:
            train_opt = pickle.load(f)

        # train_optionに追加
        train_opt.gpu_id = opt.gpu_id
        train_opt.epoch = opt.epoch
        train_opt.which_epoch = opt.which_epoch
        train_opt.resume = True

        print("load checkpoints_dir: {}".format(train_opt.checkpoints_dir))


        # set gpu ids
        if opt.gpu_id == 0:
            device = torch.device("cuda:0")
        elif opt.gpu_id == 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        train_opt.device = device

        # オプションの出力&保存
        self.print_options(train_opt)
        self.opt = train_opt

        return self.opt


if __name__ == "__main__":
    from base_options import BaseOptions

    train_options = TrainOptions()
    opt = train_options.parse()
