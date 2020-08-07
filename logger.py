# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
from torchvision.utils import save_image
import os

from utils import *

class Logger(object):
    def __init__(self, log_dir, sample_dir, load_step = 1):
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
        if load_step == 0:
            log_file = os.path.join(self.log_dir, "log.txt")
            if os.path.exists(log_file):
                os.remove(os.path.join(self.log_dir, "log.txt"))

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        save_image(
            images.data,
            os.path.join(self.sample_dir,f'step{step:0>6}.png'),
            nrow=8,
            normalize=True
        )
    
    def __call__(self, line):
        """write line to logfile"""
        print(line)
        with open(os.path.join(self.log_dir, "log.txt"), 'a') as f:
            f.write(f"{line}\n")
