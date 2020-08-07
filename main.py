import os
import argparse
from io import BytesIO
import base64

import numpy as np
from PIL import Image

from trainer import Trainer
from discriminator import Discriminator
from generator import Generator


def main(args, scope):
    G = Generator(args)
    D = Discriminator(args)
    trainer = Trainer(G, D, args)

    if args.mode == 'train':
        if args.verbose:
            trainer.show_current_model()
        try:
            trainer.train()
        finally:
            if args.poweroff:
                os.system('sudo poweroff')
    elif args.mode == 'sample':
        trainer.sample()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SN-GAN")

    # Dataset
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        choices=['CIFAR10'])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Model params
    parser.add_argument("--z_dim", default=128, type=int,
                        help="Dimension of latent vector")

    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=int, default=2e-4)
    parser.add_argument('--d_iter', type=int, default=5)
    parser.add_argument('--hinge_loss', action='store_true')

    # Misc
    parser.add_argument('--delete_old', action='store_true')
    parser.add_argument('--data_path', type=str, default='/home/ash-arch/Documents/datasets/cifar10')
    default_path = 'cifar_10_hing_loss'
    parser.add_argument('--log_path', type=str, default=f'./{default_path}/logs')
    parser.add_argument('--sample_path', type=str, default=f'./{default_path}/samples')
    parser.add_argument('--model_path', type=str, default=f'./{default_path}/models')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'])
    parser.add_argument('--nsamples', type=int, default=64)

    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--log_step', type=int, default=0)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--save_step', type=int, default=2000)
    parser.add_argument('--inception_step', type=int, default=1, help='set 0 to not calculate inception score')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'gpu'])
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--poweroff', action='store_true')
    

    args = parser.parse_args()
    if args.dataset == "CIFAR10":
        args.m_g = 4
        args.ngf = 512
        args.ndf = 512
    main(args, scope=locals())
