import torch
import random
import numpy as np
import collections
import cv2


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ResizeAB:
    def __init__(self, size, interA=cv2.INTER_NEAREST, interB=cv2.INTER_NEAREST):
        """
        Resize two images to size.

        Args:
            size (tuple): (height, weight)
            interA (cv2.INTER...): Aに適用する補完手法．
            interB (cv2.INTER...): Aに適用する補完手法．
                入力側には cv2.INTER_NEAREST がかけられる．Defaults to cv2.INTER_CUBIC.

        Returns:
            A, B
        """
        size = (size[1], size[0])  # cv2.resize のsizeの引数は(width, height)なので
        assert isinstance(size, int) or (
            isinstance(size, collections.Iterable) and len(size) == 2
        )
        self.size = size
        self.interA = interA
        self.interB = interB

    def __call__(self, A, B):
        """
        Parameter
        ----------------
        A: numpy.ndarray. ラベルマップなどの入力．
        B: numpy.ndarray．生成したい画像．
        """
        if A.shape[2] == 1:
            return (
                cv2.resize(A, self.size, interpolation=self.interA)[:, :, np.newaxis],
                cv2.resize(B, self.size, interpolation=self.interB),
            )
        else:
            return (
                cv2.resize(A, self.size, interpolation=self.interA),
                cv2.resize(B, self.size, interpolation=self.interB),
            )


class RandomHFlipAB:
    def __init__(self, p=0.5):
        """Horizontally flip the given two numpy Image randomly with a given probability.

        Args:
            p (float, optional): flip する確率. Defaults to 0.5.
        """
        self.p = p

    @staticmethod
    def hflip(img):
        """Horizontally flip the given numpy ndarray.
        Args:
            img (numpy ndarray): image to be flipped.
        Returns:
            numpy ndarray:  Horizontally flipped image.
        """
        assert _is_numpy_image(img), TypeError(
            "img should be numpy image. Got {}".format(type(img))
        )
        # img[:,::-1] is much faster, but doesn't work with torch.from_numpy()!
        if img.shape[2] == 1:
            return cv2.flip(img, 1)[:, :, np.newaxis]
        else:
            return cv2.flip(img, 1)

    def __call__(self, input_A, output_B):
        """
        Parameter
        ----------------
        input_A: numpy.ndarray
        output_B: numpy.ndarray
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return self.hflip(A), self.hflip(B)
        else:
            return A, B


class RandomCropAB:
    """Crop the given numpy ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[0:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @staticmethod
    def crop(img, i, j, h, w):
        """Crop the given PIL Image.
        Args:
            img (numpy ndarray): Image to be cropped.
            i: Upper pixel coordinate.
            j: Left pixel coordinate.
            h: Height of the cropped image.
            w: Width of the cropped image.
        Returns:
            numpy ndarray: Cropped image.
        """
        assert _is_numpy_image(img), TypeError(
            "img should be numpy image. Got {}".format(type(img))
        )

        return img[i : i + h, j : j + w, :]

    def __call__(self, A, B):
        """
        Parameter
        ----------------
        input_A: numpy.ndarray
        output_B: numpy.ndarray
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        i, j, h, w = self.get_params(A, self.size)
        return self.crop(A, i, j, h, w), self.crop(B, i, j, h, w)


class ToTensorAB:
    """Convert two images to tensor.
    """

    @staticmethod
    def to_tensor(pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
        See ``ToTensor`` for more details.
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        assert _is_numpy_image(pic), TypeError(
            "pic should be numpy image. Got {}".format(type(pic))
        )

        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor) or img.dtype == torch.uint8:
            return img.float().div(255)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def __call__(self, A, B):
        """
        Parameter
        ----------------
        input_A: numpy.ndarray
        output_B: numpy.ndarray
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        return self.to_tensor(A), self.to_tensor(B)


class ComposeAB:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, list)
        self.transforms = transforms

    def __call__(self, A, B):
        """
        transform two images.
        """
        for transform in self.transforms:
            A, B = transform(A=A, B=B)
        return A, B

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class NormalizeAB:
    def __init__(self, meanA, stdA, meanB, stdB, normA=False, normB=True):
        """
        ２つのペア画像をmean, std で正規化する

        Args:
            meanA (list): A の正規化に用いる平均値
            stdA (list): A の正規化に用いる標準偏差
            meanB (list): B の正規化に用いる平均値
            stdB (list): B の正規化に用いる標準偏差
            normA : A を正規化するかどうか
            normB : B を正規化するかどうか
        """
        self.meanA = meanA
        self.stdA = stdA
        self.meanB = meanB
        self.stdB = stdB
        self.normA = normA
        self.normB = normB

    @staticmethod
    def normalize(tensor, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor

    def __call__(self, A, B):
        """
        Args:
            A: 入力マップ
            B: 出力マップ
        """

        if self.normA:
            A = self.normalize(tensor=A, mean=self.meanA, std=self.stdA)
        if self.normB:
            B = self.normalize(tensor=B, mean=self.meanB, std=self.stdB)
        return A, B

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(meanA={0}, stdA={1}), (meanB={2}, stdB={3})".format(
                self.meanA, self.stdA, self.meanB, self.stdB
            )
        )
