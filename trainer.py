import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms

from dataloader import get_loader, GenDataset
from logger import Logger
from utils import *
from tensorboardX import SummaryWriter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Trainer:
    def __init__(self, G, D, args):
        self.train_loader, _ = get_loader(
            dataset=args.dataset,
            root=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        self.G = G
        self.D = D
        self.args = args
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.nsamples = args.nsamples
        self.d_iter = args.d_iter

        self.logger = Logger(args.log_path, args.sample_path)
        self.writer = SummaryWriter(args.log_path)
        self.model_path = args.model_path

        cuda = (args.device=='cuda' or args.device=='gpu')
        if cuda and not torch.cuda.is_available():
            self.logger("cuda not available. run in cpu mode")
            cuda = False
        self.device = 'cuda' if cuda else 'cpu'
        self.G.to(self.device)
        self.D.to(self.device)

        self.epoch = 0
        self.step = max(args.load_step, 0)
        self.old_step = self.step

        self.G_optimizer = optim.Adam(G.parameters(), self.lr, betas=(0.0, 0.9))
        self.D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), self.lr, betas=(0.0, 0.9))
        self.G_scheduler = optim.lr_scheduler.ExponentialLR(self.G_optimizer, gamma=0.99)
        self.D_scheduler = optim.lr_scheduler.ExponentialLR(self.D_optimizer, gamma=0.99)
        self.lr_scheduler = args.lr_scheduler

        self.criterion = nn.BCEWithLogitsLoss()

        self.G.apply(init_params)
        self.D.apply(init_params)
        self.fixed_z = torch.randn(self.nsamples, self.G.z_dim).to(self.device)

        if args.inception_step:
            self.inception_model = Inception(cuda=cuda)

    def train(self):
        if self.step > 0:
            self.load(f"{self.step:0>7}")
        else:
            self.sample()
        
        self.G.train()
        self.D.train()
        for self.epoch in range(1, self.args.epochs+1):
            epoch_info = self.train_epoch()

            self.logger(f"Epoch: {self.epoch:0>3} | Step: {self.step:0>8} | " +
                         " | ".join(f"{k}: {v:.5f}" for k, v in epoch_info.items()))
            
            if self.lr_scheduler:
                self.G_scheduler.step()
                self.D_scheduler.step()

            if self.args.inception_step and self.epoch % self.args.inception_step == 0:
                self.G.eval()
                self.logger('calculating inception score...')
                score_mean, score_std = self.inception_model.inception_score(GenDataset(self.G, 50000), self.batch_size, True)
                self.logger(f"Inception score at epoch {self.epoch} with 50000 generated samples - Mean: {score_mean:.3f}, Std: {score_std:.3f}")
                self.writer.add_scalars("Inception Score", {"Mean" : score_mean}, self.step)
                self.writer.add_scalars("Inception Score", {"Std" : score_std}, self.step)
                self.G.train()
        # save
        self.save(f"{self.step:0>7}")
        if self.args.delete_old and self.old_step > 0:
            os.remove(f"{self.model_path}/{self.old_step:0>7}")
        self.old_step = self.step
    
    def train_epoch(self):
        loss_dict = {'d_loss_real': 0.0, 'd_loss_fake': 0.0, 'd_loss': 0.0, 'g_loss': 0.0}
        for i, (real_imgs, real_labels) in enumerate(self.train_loader):
            self.step += 1
            real_imgs = real_imgs.to(self.device)
            real_labels = torch.ones_like(real_labels, dtype=torch.float32, device=self.device)
            fake_labels = torch.zeros_like(real_labels)
            z = torch.randn(self.batch_size, self.G.z_dim, device=self.device)

            # Discriminator
            # D = argmax{ E[logD(x)] + E[log(1-D(G(z)))] }
            self.D.zero_grad()

            d_loss_real = self.criterion(self.D(real_imgs), real_labels)

            with torch.no_grad():
                fake_imgs = self.G(z)
            d_loss_fake = self.criterion(self.D(fake_imgs), fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.D_optimizer.step()
            self.writer.add_scalars('D_loss', {"D" :d_loss.item()}, self.step)
            self.writer.add_scalars('D_loss', {"D_real" : d_loss_real.item()}, self.step)
            self.writer.add_scalars('D_loss', {"D_fake" : d_loss_fake.item()}, self.step)
            loss_dict['d_loss_real'] += d_loss_real.item()
            loss_dict['d_loss_fake'] += d_loss_fake.item()
            loss_dict['d_loss'] += d_loss.item()

            # Generator
            # G = argmax{ E[log(D(G(z)))] } (instead of argmin{ E[log(1-D(G(z)))] })
            if self.step % self.d_iter == 0:
                self.G.zero_grad()
                fake_imgs = self.G(z)

                g_loss = self.criterion(self.D(fake_imgs), real_labels)
                g_loss.backward()
                self.G_optimizer.step()
                self.writer.add_scalar('G_loss', g_loss.item(), self.step)
                loss_dict['g_loss'] += g_loss.item()

            # log
            if self.step % self.args.log_step == 0:
                self.logger(f'step: {self.step}, d_loss: {to_np(d_loss):.5f}, g_loss: {to_np(g_loss):.5f}')
            # sample image
            if self.step % self.args.sample_step == 0:
                samples = self.denorm(self.infer(self.nsamples))
                self.logger.images_summary("samples_unfixed", samples, self.step)
                        
            # save
            if self.step % self.args.save_step == 0:
                self.save(f"{self.step:0>7}")
                if self.args.delete_old and self.old_step > 0:
                    os.remove(f"{self.model_path}/{self.old_step:0>7}")
                self.old_step = self.step
        loss_dict["d_loss_real"] /= len(self.train_loader)
        loss_dict["d_loss_fake"] /= len(self.train_loader)
        loss_dict["d_loss"] /= len(self.train_loader)
        loss_dict["g_loss"] /= len(self.train_loader)//self.d_iter
        return loss_dict

    def sample(self):
        self.G.eval()
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

        with torch.no_grad():
            samples = self.denorm(self.G(self.fixed_z))
        self.logger.images_summary("samples_fixed", samples, self.step)
        self.G.train()

    def infer(self, nsamples):
        self.G.eval()
        z = torch.randn(nsamples, self.G.z_dim, device=self.device)
        with torch.no_grad():
            out = self.G(z)
        self.G.train()
        return out

    def denorm(self, x):
        # For fake data generated with tanh(x)
        x = (x + 1) / 2
        return x.clamp(0, 1)

    def show_current_model(self):
        print_network(self.G)
        print_network(self.D)

    def save(self, filename):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(
            {'G': self.G.state_dict(), 'D': self.D.state_dict(),
             'G_optimizer' : self.G_optimizer.state_dict(),
             'D_optimizer' : self.D_optimizer.state_dict(),
             'G_scheduler' : self.G_scheduler.state_dict(),
             'D_scheduler' : self.D_scheduler.state_dict()},
            os.path.join(self.model_path, filename)
        )

    def load(self, filename):
        ckpt = torch.load(os.path.join(self.model_path, filename))
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.G_optimizer.load_state_dict(ckpt['G_optimizer'])
        self.D_optimizer.load_state_dict(ckpt['D_optimizer'])
        if self.lr_scheduler:
            self.G_scheduler.load_state_dict(ckpt['G_scheduler'])
            self.D_scheduler.load_state_dict(ckpt['D_scheduler'])
