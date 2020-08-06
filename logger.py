# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
from torchvision.utils import save_image
import os

from utils import *

class Logger(object):
    def __init__(self, log_dir, sample_dir):
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        if not os.path.isdir(sample_dir):
            os.mkdir(sample_dir)

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        save_image(
            images.data,
            os.path.join(self.sample_dir,'step{step:0>6}.png'),
            nrow=8,
            normalize=True
        )
    
    def __call__(self, line):
        """write line to logfile"""
        print(line)
        with open(os.path.join(self.log_dir, "log.txt"), 'a') as f:
            f.write(f"{line}\n")
