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
        self.g_iter = args.g_iter
        self.g_losses = []
        self.d_losses = []

        self.epoch = 0
        self.step = max(args.load_step, 0)
        self.old_step = self.step

        self.logger = Logger(args.log_path)
        self.model_path = args.model_save_path

        self.G_optimizer = torch.optim.Adam(G.parameters(), self.lr, betas=(0.0, 0.9))
        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), self.lr, betas=(0.0, 0.9))
        self.G_scheduler = optim.lr_scheduler.ExponentialLR(self.G_optimizer, gamma=0.99)
        self.D_scheduler = optim.lr_scheduler.ExponentialLR(self.D_optimizer, gamma=0.99)

        self.criterion = nn.BCEWithLogitsLoss()

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

        self.G.apply(init_params)
        self.D.apply(init_params)
        self.fixed_z = to_var(torch.randn(self.nsamples, self.G.z_dim))

    def train(self):
        if self.step > 0:
            self.load(f"{self.step:0>6}")
        self.sample()
        for self.epoch in range(1, self.args.epochs+1):
            epoch_info = self.train_epoch()

            print("Epoch: %3d | Step: %8d | " % (self.epoch, self.step) +
                  " | ".join("{}: {:.5f}".format(k, v) for k, v in epoch_info.items()))
            #self.sample()
            self.G_scheduler.step()
            self.D_scheduler.step()

        if self.args.inception_score:
            score_mean, score_std = inception_score(GenDataset(self.G, 50000), torch.cuda.is_available(), self.batch_size, True)
            print("Inception score at epoch {} with 50000 generated samples - Mean: {:.3f}, Std: {:.3f}".format(self.epoch, score_mean, score_std))            
    
    def train_epoch(self):
        self.G.train()
        self.D.train()

        for i, (real_imgs, real_labels) in enumerate(self.train_loader):
            real_imgs, real_labels = to_var(real_imgs), to_var(real_labels)
            self.step += 1
            d_loss_ = 0.0
            for _ in range(self.d_iter):
                # Discriminator
                # V(D) = E[logD(x)] + E[log(1-D(G(z)))]
                self.D.zero_grad()
                z = to_var(torch.randn(self.batch_size, self.G.z_dim))

                real_labels.fill_(1)
                real_labels = real_labels.float()

                d_loss_real = self.criterion(self.D(real_imgs), real_labels)
                fake_imgs = self.G(z).detach()
                fake_labels = real_labels.clone()
                fake_labels.fill_(0)
                d_loss_fake = self.criterion(self.D(fake_imgs), fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_loss_ += d_loss.item()

                self.D_optimizer.step()
            d_loss_ /= self.d_iter
            self.d_losses.append(d_loss_)

            # Generator
            # V(G) = -E[log(D(G(z)))]
            g_loss_ = 0.0
            for _ in range(self.g_iter):
                self.G.zero_grad()
                fake_imgs = self.G(z)
                fake_labels.fill_(1)

                g_loss = self.criterion(self.D(fake_imgs), fake_labels)
                g_loss.backward()

                self.G_optimizer.step()
                g_loss_ += g_loss.item()
            g_loss_ /= self.g_iter
            self.g_losses.append(g_loss_)

            # log
            if self.step % self.args.log_step == 0:
                print('step: {}, d_loss: {:.5f}, g_loss: {:.5f}'.format(self.step, to_np(d_loss), to_np(g_loss)))
            # sample image
            if self.step % self.args.sample_step == 0:
                samples = self.denorm(self.infer(self.nsamples))
                self.logger.images_summary("samples_unfixed", samples, self.step)
                        
            # save
            if self.step % self.args.save_step == 0 or i == len(self.train_loader) - 1:
                self.save(f"{self.step:0>6}")
                if self.args.delete_old and self.old_step > 0:
                    os.remove(f"{self.model_path}/{self.old_step:0>6}")
                    self.old_step = self.step
            
        return {'d_loss_real': to_np(d_loss_real), 'd_loss_fake': to_np(d_loss_fake),
                'd_loss': to_np(d_loss), 'g_loss': to_np(g_loss)}

    def sample(self):
        self.G.eval()
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

        samples = self.denorm(self.G(self.fixed_z))
        self.logger.images_summary("samples_fixed", samples, self.step)

    def infer(self, nsamples):
        self.G.eval()
        z = to_var(torch.randn(nsamples, self.G.z_dim))
        return self.G(z)

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
            {'G': self.G.state_dict(), 'D': self.D.state_dict()},
            os.path.join(self.model_path, filename)
        )
        np.save(os.path.join(self.model_path, 'd_losses'),np.array(self.d_losses))
        np.save(os.path.join(self.model_path, 'g_losses'),np.array(self.g_losses))

    def load(self, filename):
        ckpt = torch.load(os.path.join(self.args.model_save_path, filename))
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.d_losses = list(np.load(os.path.join(self.model_path, 'd_losses.npy')))
        self.g_losses = list(np.load(os.path.join(self.model_path, 'g_losses.npy')))
