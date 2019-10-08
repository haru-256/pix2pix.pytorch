import torch

# import torch.nn as nn
import torchvision
import math
import matplotlib.pyplot as plt
import re
import pathlib


class BaseModel(object):
    """ Image2Imageの基クラス
    実装すべきメソッドは以下の通り．
        __init__ : Generator, Discriminatorのモデルの構築，OptimizerとLossの定義．
        forward : 生成画像とロスのディクショナリを返すメソッド．
    """

    def __init__(self, opt):
        self.opt = opt
        self.netG = None
        self.netD = None

    @staticmethod
    def migrate(data_dict, device, verbose=False):
        """migrate data to device(e.g. cuda)

        Args:
            data_dict (dict): each keys is "real_image", "input_map".

        Returns:
            data_dict (dict): arguments(data_dict) after migrating.
        """
        for key, data in data_dict.items():
            if not isinstance(data, torch.Tensor):
                if verbose:
                    print("{} はGPUへ転送できない".format(key))
                continue
            data_dict[key] = data.to(device)
        return data_dict

    def save_gen_images(self, epoch, dataloaders4vis):
        """学習データ＆検証データを生成し，画像を保存するメソッド

        Args:
            epoch (int): epoch
            dataloaders (dict): dataloader のディクショナリ．key は"train", "val"
        """
        self.netG.train()  # 多様性のため推論時もdropout を適用．しかし効果は薄い．詳細は論文参照．

        for phase in ["train", "val"]:
            # 学習データについての生成画像を保存
            with torch.no_grad():
                data_dict = next(iter(dataloaders4vis[phase]))
                # onehot_segmapに変換 & migrate to gpu
                data_dict = self.migrate(data_dict, self.opt.device)

                # Generate fake image
                fake_B = self.netG(data_dict["A"]).cpu()
                assert bool(
                    ((-1 <= fake_B) * (fake_B <= 1)).all()
                ), "generated image data range is not from -1 to 1. Got: {}".format(
                    (fake_B.min(), fake_B.max())
                )

            total = fake_B.shape[0]
            assert (
                total == self.opt.vis_num
            ), "可視化するデータ数とあっていない. Got total:{}, vis_num:{}".format(
                total, self.opt.vis_num
            )
            ncol = int(math.sqrt(total))
            nrow = math.ceil(float(total) / ncol)
            images = torchvision.utils.make_grid(
                fake_B, normalize=False, nrow=nrow, padding=1
            )
            images = images * self.opt.stdB + self.opt.meanB
            assert bool(
                ((0 <= images) * (images <= 1)).all()
            ), "Image is not from 0 to 1. Got: {}".format((images.min(), images.max()))
            plt.figure(figsize=(8, 8))
            plt.imshow(images.numpy().transpose(1, 2, 0))
            plt.axis("off")
            plt.title("Epoch: {}".format(epoch), fontsize=15)
            plt.tight_layout()
            path = self.opt.image_dir / "{0}/epoch{1:0>4}.png".format(phase, epoch)
            if not path.parent.exists():
                path.parent.mkdir()
            plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
            plt.close()

        self.netG.train()  # 学習フェーズ

    def get_latest_file(self):
        """
        latestのepochを探し，そのepoch時のファイルパスを返す．
        また，start_epochも設定する．

        Args:
            file_name (str, optional): [description]. Defaults to "pix2pix_{}epoch.tar".

        Returns:
            latest_file : 最新epochが入ったファイルパス．
        """
        pattern1 = "[0-9]+epoch"
        pattern1 = re.compile(pattern1)
        pattern2 = "epoch"
        pattern2 = re.compile(pattern2)
        latest = 0
        path_list = self.opt.model_dir.glob("*")
        for path in path_list:
            file_name = path.stem
            epoch_name = re.findall(pattern1, file_name)
            assert len(epoch_name) == 1, "正しいファイル名出ない．Got {}".format(epoch_name)
            epoch = int(re.sub(pattern2, "", epoch_name[0]))
            if epoch > latest:
                latest = epoch
        latest_file = pathlib.Path(
            self.opt.model_dir / "pix2pix_{}epoch.tar".format(latest)
        )
        assert latest_file.exists(), "{} is Not Founed".format(latest_file)
        return latest_file, latest
