# coding=utf-8
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from .base_model import BaseModel
import sys
import networks
import pdb


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class InpaintModel(BaseModel):
    def __init__(self, opt):
        BaseModel.initialize(self, opt)
        self.name = opt.model + '_' + opt.base
        self.v_mean = self.Tensor(opt.mean)[None, ..., None, None]
        self.v_std = self.Tensor(opt.std)[None, ..., None, None]
        net = networks.InpaintGenerator()
        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()

        dis_net = networks.InpaintDiscriminator(in_channels=3, use_sigmoid=True)
        dis_net = torch.nn.parallel.DataParallel(dis_net)
        self.dis_net = dis_net.cuda()

        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                 opt.imageSize, opt.imageSize)
        self.targets = self.Tensor(opt.batchSize, opt.input_nc,
                                 opt.imageSize, opt.imageSize)
        self.mask = self.Tensor(opt.batchSize, 1,
                                 opt.imageSize, opt.imageSize)
        self.prediction = None
        self.loss = {}

        if opt.phase is 'test':
            pass
            # print("===========================================LOADING parameters====================================================")
            # model_parameters = self.load_network(model, 'G', 'best_vanila')
            # model.load_state_dict(model_parameters)
        else:
            self.criterion_l1 = nn.L1Loss(reduce=False)
            self.criterion_adv = AdversarialLoss(type='nsgan').cuda()
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.dis_optimizer = torch.optim.Adam(self.dis_net.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def save(self, label):
        self.save_network(self.net, self.name+'_g', label)
        self.save_network(self.dis_net, self.name+'_d', label)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name+'_g', label)
        self.load_network(self.dis_net, self.name+'_d', label)

    def show_tensorboard_eval(self, num_iter):
        for k, v in self.performance.items():
            self.writer.add_scalar(k, v, num_iter)

    def show_tensorboard(self, num_iter, num_show=4):
        for k, v in self.loss.items():
            self.writer.add_scalar(k, v, num_iter)
        num_show = num_show if self.input.size(0) > num_show else self.input.size(0)

        pred = self.prediction[:num_show] * self.mask[:num_show] + self.input[:num_show]
        self.writer.add_image('prediction', torchvision.utils.make_grid(pred.detach()), num_iter)

        img = self.input[:num_show]
        self.writer.add_image('image', torchvision.utils.make_grid(img), num_iter)

    def set_input(self, image, mask):
        image = image.cuda()
        mask = mask.unsqueeze(1).cuda()
        self.input.resize_(image.size()).copy_(image*(1-mask))
        self.targets.resize_(image.size()).copy_(image)
        self.mask.resize_(mask.size()).copy_(mask)

    def forward(self):
        # print("We are Forwarding !!")
        self.prediction = self.net.forward(self.input, self.mask)

    def test(self, image, mask, name, WW, HH):
        self.set_input(image, mask)
        b, c, h, w = image.shape
        with torch.no_grad():
            self.forward()
            outputs = self.prediction * 255
            targets = self.targets * 255
            mse = ((outputs-targets)**2*self.mask).sum(3).sum(2).sum(1)
            mse /= (c*h*w)
            psnr = 10 * np.log10(255.0*255.0 / mse)
        return psnr.sum().item()
        # outputs = outputs.detach().cpu().numpy() * 255
        # outputs = outputs.transpose((0, 2, 3, 1))
        # for ii, msk in enumerate(outputs):
        #     msk = Image.fromarray(msk.astype(np.uint8))
        #     msk = msk.resize((WW[ii], HH[ii]))
        #     msk.save('{}/{}.jpg'.format(self.opt.results_dir, name[ii]), 'PNG')


    def backward(self):
        # discriminator loss
        # dis_input_real = self.targets
        # dis_input_fake = self.prediction.detach()
        # dis_real, _ = self.dis_net(dis_input_real)  # in: [rgb(3)]
        # dis_fake, _ = self.dis_net(dis_input_fake)  # in: [rgb(3)]
        # dis_real_loss = self.criterion_adv(dis_real, True, True)
        # dis_fake_loss = self.criterion_adv(dis_fake, False, True)
        # dis_loss = (dis_real_loss + dis_fake_loss) / 2
        # dis_loss.backward()
        # self.loss['dis'] = dis_loss.item()
        #
        # gen_fake, _ = self.dis_net(self.prediction)  # in: [rgb(3)]
        # gen_gan_loss = self.criterion_adv(gen_fake, True, False) * 0.1
        # self.loss['gen'] = gen_gan_loss.item()

        # Combined loss
        loss_var = self.criterion_l1(self.prediction, self.targets)
        loss_var = (loss_var * self.mask).mean()
        self.loss['l1'] = loss_var.item()
        loss_gen = loss_var
        # loss_gen = gen_gan_loss + loss_var
        loss_gen.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

