# coding=utf-8
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from torch.autograd import Variable
from .base_model import _BaseModel
import sys
from evaluate_sal import fm_and_mae
from tensorboardX import SummaryWriter
from datetime import datetime
from .dcgan import Generator, Discriminator
import pdb

thismodule = sys.modules[__name__]


class GANModel(_BaseModel):
    def __init__(self, opt, nz, ngf, ndf):
        _BaseModel.initialize(self, opt)
        self.name = 'dcgan_' + opt.base
        self.v_mean = self.Tensor(opt.mean)[None, ..., None, None]
        self.v_std = self.Tensor(opt.std)[None, ..., None, None]
        ngpu = len(opt.gpu_ids)
        nc = opt.input_nc
        self.nz = nz

        net_g = Generator(ngpu, nz, ngf, nc)
        net_d = Discriminator(ngpu, nc, ndf)
        self.net_g = torch.nn.parallel.DataParallel(net_g).cuda()
        self.net_d = torch.nn.parallel.DataParallel(net_d).cuda()

        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                 opt.imageSize, opt.imageSize)
        self.fixed_noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()
        self.real_label = 1
        self.fake_label = 0

        if opt.phase is 'test':
            pass
            #print("===========================================LOADING parameters====================================================")
            # model_parameters = self.load_network(model, 'G', 'best_vanila')
            # model.load_state_dict(model_parameters)
            #net_g.load_state_dict(torch.load('/home/zhang/segggFiles/pbr-mlt/_99000_net_dcgan_densenet169_g.pth'))

        else:
            self.criterion = nn.BCELoss()
            self.optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    def save(self, label):
        self.save_network(self.net_g, self.name+'_g', label, self.gpu_ids)
        self.save_network(self.net_d, self.name+'_d', label, self.gpu_ids)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net_g, self.name+'_g', label, self.gpu_ids)
        self.load_network(self.net_d, self.name+'_d', label, self.gpu_ids)

    def show_tensorboard(self, num_iter, num_show=4):
        self.writer.add_scalar('errD', self.errD, num_iter)
        self.writer.add_scalar('errG', self.errG, num_iter)
        self.writer.add_scalar('D_x', self.D_x, num_iter)
        self.writer.add_scalar('D_G_z1', self.D_G_z1, num_iter)
        self.writer.add_scalar('D_G_z2', self.D_G_z2, num_iter)
        num_show = num_show if self.input.size(0) > num_show else self.input.size(0)

        img = self.fake.detach()[:num_show] * self.v_std + self.v_mean
        self.writer.add_image('gen', torchvision.utils.make_grid(img), num_iter)

        img = self.input[:num_show]*self.v_std + self.v_mean
        self.writer.add_image('image', torchvision.utils.make_grid(img), num_iter)

    def set_input(self, input):
        self.input.resize_(input.size()).copy_(input.cuda())


    def test(self, i=None):
        with torch.no_grad():
            noise = torch.randn(self.opt.batchSize, self.nz, 1, 1).cuda()
            fake = self.net_g(noise)
            fake = fake* self.v_std + self.v_mean
        outputs = fake.detach().cpu().numpy() * 255
        outputs = outputs.transpose((0, 2, 3, 1))
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            if i :
                msk.save('{}/{}_{}.jpg'.format(self.opt.results_dir, i, ii), 'JPEG')
            else:
                msk.save('{}/{}.jpg'.format(self.opt.results_dir, ii), 'JPEG')


    def optimize_parameters_d(self):
        self.net_d.zero_grad()
        batch_size = self.input.size(0)
        label = torch.Tensor(batch_size).cuda().fill_(self.real_label)

        output = self.net_d(self.input)
        errD_real_var = self.criterion(output, label)
        errD_real_var.backward()

        self.D_x = output.mean().item()

        noise = torch.randn(batch_size, self.nz, 1, 1).cuda()
        fake = self.net_g(noise)
        self.fake = fake
        label.fill_(self.fake_label)
        output = self.net_d(fake.detach())
        errD_fake_var = self.criterion(output, label)
        errD_fake_var.backward()

        self.D_G_z1 = output.mean().item()

        self.errD = errD_real_var.item() + errD_fake_var.item()

        self.optimizer_d.step()

    def optimize_parameters_g(self):
        self.net_g.zero_grad()
        label = torch.Tensor(self.input.size(0)).cuda().fill_(self.real_label)
        output = self.net_d(self.fake)
        errG_var = self.criterion(output, label)
        errG_var.backward()
        self.D_G_z2 = output.mean().item()
        self.errG = errG_var.item()
        self.optimizer_g.step()


    def switch_to_train(self):
        self.net_g.train()
        self.net_d.train()

    def switch_to_eval(self):
        self.net_g.eval()
        self.net_d.eval()

