import numpy as np
import cv2


class SplitImage(object):
    """Split the image to separete it into input:A, output:B

    Parameers
    --------------
    location: int
        location of boundary line to separete image into input:A, output:B
    """

    def __init__(self, location):
        self.location = location

    def __call__(self, sample):
        # In case of tuple, suppose (images, labels)
        if isinstance(sample, tuple):
            images, _ = sample
