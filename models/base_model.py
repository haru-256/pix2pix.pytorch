import torch

# import torch.nn as nn
import torchvision
import math
import matplotlib.pyplot as plt
import re
import pathlib


class BaseModel(object):
    """ Image2Imageの抽象基底クラス
    実装すべきメソッドは以下の通り．
        __init__ : Generator, Discriminatorのモデルの構築，OptimizerとLossの定義．
        forward : 生成画像とロスのディクショナリを返すメソッド．
    """

    def __init__(self, opt):
        # super(BaseModel, self).__init__()
        self.opt = opt
        self.netG = None
        self.netD = None

    def migrate(self, data_dict, verbose=False):
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
            data_dict[key] = data.to(self.opt.device)
        return data_dict

        def save_gen_images(self, epoch, dataloaders4vis, writer=None):
            """学習データ＆検証データの生成画像を保存するメソッド

            Args:
                epoch (int): epoch
                dataloaders (dict): dataloader のディクショナリ．key は"train", "val"
                writer (SummaryWriter): 生成画像をadd するためのwriter
            """
            self.netG.train()  # 多様性のため推論時もdropout を適用．しかし効果は薄い．詳細は論文参照．

            for phase in ["train", "val"]:
                # 学習データについての生成画像を保存
                with torch.no_grad():
                    data_dict = next(iter(dataloaders4vis[phase]))
                    # onehot_segmapに変換 & migrate to gpu
                    data_dict = self.migrate(data_dict)

                    # Generate fake image
                    z_map = self.netE(
                        data_dict["real_image"],
                        segmap=data_dict["input_map"],
                        oneHot_segmap=data_dict["oneHot"],
                    )
                    fake_images = self.netG(
                        input_map=data_dict["oneHot"], z_map=z_map
                    ).cpu()
                    assert bool(
                        ((-1 <= fake_images) * (fake_images <= 1)).all()
                    ), "generated image data range is not from -1 to 1. Got: {}".format(
                        (fake_images.min(), fake_images.max())
                    )

                total = fake_images.shape[0]
                assert (
                    total == self.opt.vis_num
                ), "可視化するデータ数があっていない. Got total:{}, vis_num:{}".format(
                    total, self.opt.vis_num
                )
                ncol = int(math.sqrt(total))
                nrow = math.ceil(float(total) / ncol)
                images = torchvision.utils.make_grid(
                    fake_images, normalize=False, nrow=nrow, padding=1
                )
                images = images * self.opt.std + self.opt.mean
                assert bool(
                    ((0 <= images) * (images <= 1)).all()
                ), "Image is not from 0 to 1. Got: {}".format(
                    (images.min(), images.max())
                )
                # sumarry に追加
                if writer is not None:
                    writer.add_image(
                        "{0}/epoch{1:0>4}".format(phase, epoch), images, epoch
                    )
                plt.figure(figsize=(12, 9))
                plt.imshow(images.numpy().transpose(1, 2, 0))
                plt.axis("off")
                plt.title("Epoch: {}".format(epoch), fontsize="15")
                plt.tight_layout()
                path = self.opt.image_dir / "{0}/epoch{1:0>4}.png".format(phase, epoch)
                if not path.parent.exists():
                    path.parent.mkdir()
                plt.savefig(path, bbox_inches="tight", pad_inches=0.0, dpi=200)
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
            self.opt.model_dir / "{}_{}epoch.tar".format(self.opt.model, latest)
        )
        assert latest_file.exists(), "{} is Not Founed".format(latest_file)
        self.opt.start_epoch = latest + 1
        return latest_file, latest