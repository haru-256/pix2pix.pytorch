import numpy as np
import cv2
import torch


class SplitImage(object):
    """Split the image(torch.Tensor) to separete it into input:A, output:B

    Parameers
    --------------
    location: int
        location of boundary line to separete image into input:A, output:B
    """

    def __init__(self, location):
        self.location = location

    def __call__(self, sample):
        """Split torch.Tensor

        Paramaeters
        ----------------
        sample: torch.Tensor
            torch.Tensor that represents images, of which format is (N, C, H, W)
        """
        # In case of tuple, suppose (images, labels)
        if isinstance(sample, tuple):
            images, _ = sample
        else:
            images = sample

        assert isinstance(
            images, torch.Tensor), "inputs image is not torch.Tensor. Got {}".format(type(images))

        _, _, _, w = images.shape

        inputs_A = images[:, :, 0:int(w/2), :]
        outputs_B = images[:, :, int(w/2)+1:w, :]

        return inputs_A, outputs_B
