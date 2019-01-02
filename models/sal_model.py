# coding=utf-8
import pdb
import numpy as np
import torch
import torch.nn as nn
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
from fcn import FCN
from deeplab import DeepLab
from unet import UNet
import pdb

thismodule = sys.modules[__name__]


class SalModel(_BaseModel):
    def __init__(self, opt):
        _BaseModel.initialize(self, opt)
        self.name = opt.model + '_' + opt.base
        self.v_mean = self.Tensor(opt.mean)[None, ..., None, None]
        self.v_std = self.Tensor(opt.std)[None, ..., None, None]
        net = getattr(thismodule, opt.model)(pretrained=opt.isTrain and (not opt.from_scratch),
                                                      c_output=1,
                                                      base=opt.base)

        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()

        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                 opt.imageSize, opt.imageSize)

        if opt.phase is 'test':
            pass
            # print("===========================================LOADING parameters====================================================")
            # model_parameters = self.load_network(model, 'G', 'best_vanila')
            # model.load_state_dict(model_parameters)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr)

    def save(self, label):
        self.save_network(self.net, self.name, label, self.gpu_ids)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name, label, self.gpu_ids)

    def show_tensorboard(self, num_iter, num_show=4):
        self.writer.add_scalar('loss', self.loss, num_iter)
        num_show = num_show if self.input.size(0) > num_show else self.input.size(0)

        pred = F.sigmoid(self.prediction[:num_show])
        pred = pred[:, None, ...]
        self.writer.add_image('prediction', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)

        img = self.input[:num_show]*self.v_std + self.v_mean
        self.writer.add_image('image', torchvision.utils.make_grid(img), num_iter)

    def set_input(self, input, targets=None):
        self.input.resize_(input.size()).copy_(input.cuda())
        self.targets = targets
        if targets is not None:
            self.targets = self.targets.cuda()


    def forward(self):
        # print("We are Forwarding !!")
        self.prediction = self.net.forward(self.input)
        self.prediction = self.prediction.squeeze(1)


    def test(self, input, name, WW, HH):
        self.set_input(input)
        with torch.no_grad():
            self.forward()
            outputs = F.sigmoid(self.prediction)
        outputs = outputs.detach().cpu().numpy() * 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')


    def backward(self):
        # Combined loss
        self.loss_var = self.criterion(self.prediction, self.targets)
        self.loss_var.backward()
        self.loss = self.loss_var.data[0]


    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

