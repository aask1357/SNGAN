# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
from torchvision.utils import save_image

from utils import *

class Logger(object):
    def __init__(self, log_dir=None):
        self.log_dir = log_dir

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        save_image(
            images.data,
            f'{self.log_dir}/step{step:0>6}.png',
            nrow=8,
            normalize=True
        )
