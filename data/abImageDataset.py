import cv2
import torch
from torch.utils.data import Dataset
import pathlib
import pandas as pd
from .opencv_transforms import (
    ComposeAB,
    ToTensorAB,
    NormalizeAB,
    RandomHFlipAB,
    RandomCropAB,
    ResizeAB,
)


class ABImageDataset(Dataset):
    """
    A(input), B(output) image dataset.

    Args:
        df_path (pathlib.PosixPath or str) : columns=["A_path", "B_path"] のcsv ファイルへのパス
        opt (Namespace) : プログラムの引数
        mode_inference (boolean) : True だと前処理（例えばrandom Flip，random crop など）を行わない
        vis (boolean) : True だとopt.vis_num個のデータだけ扱う．学習途中に画像を生成するときに使用．
    """

    def __init__(self, df_path, opt, mode_inference=False, vis=False):
        if isinstance(df_path, str):
            df_path = pathlib.Path(df_path)
        assert df_path.exists()

        # csv file を読みこむ
        if vis:
            df = pd.read_csv(df_path).iloc[: opt.vis_num]  # 可視化用
        else:
            df = pd.read_csv(df_path)  # read csv file

        # パスを取得
        self.A_path = df["A_path"]
        self.B_path = df["B_path"]
        self.opt = opt
        self.params = get_params(
            opt=opt,
            size=(self.opt.img_height, self.opt.img_width),
            mode_inference=mode_inference,
        )
        assert not (
            mode_inference and self.params["flip"]
        ), "do not flip in case of phase == validation"
        # transformar
        self.transformar = get_transform(opt, self.params)

    def __getitem__(self, idx):

        # 入力（A）を読み込む
        if self.opt.A_nc == 1:
            A = cv2.imread(self.A_path.iloc[idx])[:, :, 0:1]
        else:
            A = cv2.cvtColor(cv2.imread(self.A_path.iloc[idx]), cv2.COLOR_BGR2RGB)
        # 出力（B）を読み込む
        if self.opt.B_nc == 1:
            B = cv2.imread(self.B_path.iloc[idx])[:, :, 0:1]
        else:
            B = cv2.cvtColor(cv2.imread(self.B_path.iloc[idx]), cv2.COLOR_BGR2RGB)

        # データ拡張
        A, B = self.transform(A=A, B=B)

        # 値をチェック
        assert (not self.opt.normA) or torch.all(
            ((0.0 - self.opt.meanA) / self.opt.stdA <= A)
            * (A <= (1.0 - self.opt.meanA) / self.opt.stdA)
        )
        assert (not self.opt.normB) or torch.all(
            ((0.0 - self.opt.meanB) / self.opt.stdB <= B)
            * (B <= (1.0 - self.opt.meanB) / self.opt.stdB)
        )

        return {
            "A": A,
            "B": B,
            "A_path": self.A_path.iloc[idx],
            "B_path": self.B_path.iloc[idx],
        }

    def __len__(self):
        return len(self.data_path)


class Edges2Shoes(ABImageDataset):
    def __getitem__(self, idx):
        # 入力（edge）を読み込む
        edge = cv2.imread(self.A_path.iloc[idx])[:, :, 0:1]
        # 出力（B）を読み込む
        img = cv2.cvtColor(cv2.imread(self.B_path.iloc[idx]), cv2.COLOR_BGR2RGB)

        # データ拡張
        edge, img = self.transform(A=edge, B=img)

        return {
            "A": edge,
            "B": img,
            "A_path": self.A_path.iloc[idx],
            "B_path": self.B_path.iloc[idx],
        }


_cv2_interpolation_to_str = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def get_transform(opt, params):
    """get transforms

    Args:
        opt (argparse): プログラムの引数
        params (dict): return of function 'get_params'
    """
    transform_list = []
    if "scale_width" in opt.resize_or_crop:
        transform_list.append(
            ResizeAB(
                size=params["scale_size"],
                interA=_cv2_interpolation_to_str[opt.interA],
                interB=_cv2_interpolation_to_str[opt.interB],
            )
        )

    if "crop" in opt.resize_or_crop:
        transform_list.append(RandomCropAB(size=params["crop_size"]))

    if params["flip"]:
        transform_list.append(RandomHFlipAB(p=opt.flip_p))

    transform_list += [
        ToTensorAB(),
        NormalizeAB(
            meanA=[opt.meanA] * opt.A_nc,
            stdA=[opt.stdA] * opt.A_nc,
            meanB=[opt.meanB] * opt.B_nc,
            stdB=[opt.stdB] * opt.B_nc,
            normA=opt.normA,
            normB=opt.normB,
        ),
    ]

    return ComposeAB(transform_list)


def get_params(opt, size, mode_inference=False):
    """get parameters for preprocess

    Args:
        opt (argparser): arguments of this program
        size (tuple): original size of images
        phase (string): phase of process

    Returns:
        params (dictionary): dictionary of parameters.
            Each key is "crop_pos", "scale_width_and_crop"
    """
    w, h = size
    if "scale_width" in opt.preprocess:
        new_w = opt.scaleSize
        new_h = opt.scaleSize * h // w
    else:
        new_h = h
        new_w = w

    if "crop" in opt.preprocess:
        crop_h = opt.cropSize
        crop_w = opt.cropSize * h // w
    else:
        crop_h = None
        crop_w = None

    if opt.no_flip or (mode_inference == True):
        flip = False
    else:
        flip = True

    return {"scale_size": (new_h, new_w), "crop_size": (crop_h, crop_w), "flip": flip}

