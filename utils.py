import random
import math
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import torchvision
from opencv_transforms import Resize, RandomHorizontalFlip, ToTensor, Normalize
import opencv_functional as F


def plot_loss(log, path, colors=["tab:red", 'mediumblue'], markers=['o', 'x'], ms=10):
    """
    plot both generator loss and  discriminator loss

    Parameters
    ------------------
    log: dict
        dictionary hat contains generator loss, discriminator loss.

    path: pathlib.Path
        save file path
    """
    dis_loss = [log[key]['train_dis_loss'] for key in log.keys()]
    gen_loss = [log[key]['train_gen_loss'] for key in log.keys()]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    _ = ax.plot(gen_loss, label='gen loss',
                c=colors[0], marker=markers[0], ms=ms)
    ax.grid(axis="y")
    ax.set_xlim([-0.8, 15])
    ax2 = ax.twinx()
    _ = ax2.plot(dis_loss, label='dis loss',
                 c=colors[0], marker=markers[0], ms=ms)
    fig.legend(loc='upper right', bbox_to_anchor=(1, 0.5),
               bbox_transform=ax.transAxes, frameon=True, shadow=True, fontsize=17)
    ax.tick_params(labelsize=13)
    ax2.tick_params(labelsize=13)
    ax.set_xticklabels([2*i+1 for i in range(-1, len(dis_loss))])
    ax.set_xlabel("epoch", fontsize=18, labelpad=13)
    ax.set_ylabel("Gen Loss", fontsize=18, labelpad=13)
    ax2.set_ylabel("Dis Loss", fontsize=18, labelpad=13)
    plt.grid()
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)


def visualize(epoch, gen, val_dataloader, log_dir=None, device=None,
              mean=127.5, std=127.5):
    """
    visualize generator images
    Parmameters
    -------------------
    epoch: int
        number of epochs

    gen: torch.nn.Module
        generator model

    val_dataloader: torch.utils.data.DataLoader
        dataloader for val 

    log_dir: pathlib.Path
        path to output directory

    device: torch.device

    mean: float
        mean that is used to normalize inputs data.

    std: float
        std that is used to normalize inputs data.
    """
    gen.train()  # apply Dropout and BatchNorm during inference as well

    with torch.no_grad():
        for inputs, _ in val_dataloader:
            fake_outputs = gen(inputs.to(device)).cpu()

    total = fake_outputs.shape[0]
    ncol = int(math.sqrt(total))
    nrow = math.ceil(float(total)/ncol)
    images = torchvision.utils.make_grid(
        fake_outputs, normalize=False, nrow=nrow, padding=1)
    images = images * std + mean
    assert ((-1 <= images) * (images <= 1)
            ).all(), "plot data range"
    " is not from 0 to 1. Got: {}".format(
        (images.min(), images.max()))
    plt.imshow(images.numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.title("Epoch: {}".format(epoch))
    plt.tight_layout()
    plt.savefig(log_dir / "epoch{:0>4}.png".format(epoch),
                bbox_inches="tight", pad_inches=0.05)


class ComposeTwoIMG(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_A, input_B):
        """
        transform two images.
        """
        for t in self.transforms:
            input_A, input_B = t(input_A, input_B)
        return input_A, input_B


class SplitImage(object):
    """Split the image into half, to separete it into input:A, output

    Parameter:
    ----------------------
    right_is_A: bool
        whether right is input image A.
    """

    def __init__(self, right_is_A=True):
        self.right_is_A = right_is_A

    def __call__(self, sample):
        """Split torch.Tensor

        Paramaeters
        ----------------
        sample: ndarray
            image of which format is (H, W, C)
        """
        _, w, _ = sample.shape

        assert isinstance(
            sample, np.ndarray), "inputs image is not np.ndarray. Got {}".format(type(sample))
        if self.right_is_A:
            return sample[:, int(w/2):, :], sample[:, 0:int(w/2), :]
        else:
            return sample[:, 0:int(w/2), :], sample[:, int(w/2):, :]


class ResizeTwoIMG(Resize):
    """
    Resize two images to size.
    """

    def __call__(self, input_A, output_B):
        """
        Parameter
        ----------------
        input_A: numpy.ndarray

        output_B: numpy.ndarray
        """

        return F.resize(input_A, self.size, self.interpolation), F.resize(output_B, self.size, self.interpolation)


class RandHFlipTwoIMG(RandomHorizontalFlip):
    """Horizontally flip the given two numpy Image randomly with a given probability.
    """

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
            return F.hflip(input_A), F.hflip(output_B)
        return input_A, output_B


class RandomCropTwoIMG(object):
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

    def __call__(self, input_A, output_B):
        """
        Parameter
        ----------------
        input_A: numpy.ndarray

        output_B: numpy.ndarray

        Returns:
            numpy ndarray: Randomly flipped image.
        """
        # assert input_A.shape != output_B.shape, "input:A size is not same as output_B. input_A:{} output_B:{}".format(
        #     input_A.shape, output_B.shape)

        i, j, h, w = self.get_params(input_A, self.size)

        return F.crop(input_A, i, j, h, w), F.crop(output_B, i, j, h, w)


class ToTensorTwoIMG(ToTensor):
    """Convert a two images to tensor.
    """

    def __call__(self, input_A, output_B):
        """
        Parameter
        ----------------
        input_A: numpy.ndarray

        output_B: numpy.ndarray

        Returns:
            numpy ndarray: Randomly flipped image.
        """
        return F.to_tensor(input_A), F.to_tensor(output_B)


def default_loader(path):
    # return cv2.imread(str(path))
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


class ABImageDataset(Dataset):
    """
    A(simpler), B(more complicated) image dataset.

    Parameters
    -----------------------------
    root: pathlib.PosixPath
        data dir

    transfor: torchvision.transform.Compose
    """

    def __init__(self, root, transform=None, loader=default_loader,
                 spliter=SplitImage(), normalizer=Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                 val_size=None):
        if not root.is_absolute():
            self.abs_data_dir = root.resolve()
        else:
            self.abs_data_dir = root
        if val_size:
            self.data_path = [path for i, path in enumerate(
                self.abs_data_dir.glob("*.jpg")) if i < val_size]
        else:
            self.data_path = [path for path in self.abs_data_dir.glob("*.jpg")]
        self.loader = loader
        self.spliter = spliter
        self.normalizer = normalizer
        self.transform = transform

    def __getitem__(self, idx):
        image = self.loader(self.data_path[idx])
        input_A, output_B = self.spliter(image)

        if self.transform:
            input_A, output_B = self.transform(input_A, output_B)

        # apply normalization to output_B only
        if self.normalizer:
            return input_A, self.normalizer(output_B)
        return input_A, output_B

    def __len__(self):
        return len(self.data_path)
