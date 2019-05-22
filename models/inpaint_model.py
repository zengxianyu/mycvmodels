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
from . import networks
import random
import pdb


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1.0 - prediction).mean()
            else:
                loss = F.relu(1.0 + prediction).mean()
        else:
            raise NotImplementedError

        return loss


class InpaintModel(BaseModel):
    def __init__(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.name = opt.model + '_' + opt.base
        self.w_adv = 0.1
        self.v_mean = self.Tensor(opt.mean)[None, ..., None, None]
        self.v_std = self.Tensor(opt.std)[None, ..., None, None]
        net = getattr(networks, opt.model).InpaintGenerator()
        # net = networks.contextattention.InpaintGenerator()
        # net = networks.selfattention.InpaintGenerator()
        # net = networks.woattention.InpaintGenerator()
        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()

        dis_net = getattr(networks, opt.model).InpaintDiscriminator()
        # dis_net = networks.contextattention.InpaintDiscriminator()
        # dis_net = networks.selfattention.InpaintDiscriminator()
        # dis_net = networks.woattention.InpaintDiscriminator()
        dis_net = torch.nn.parallel.DataParallel(dis_net)
        self.dis_net = dis_net.cuda()

        self.input = None
        self.targets = None
        self.mask = None
        self.prediction = None
        self.loss = {}

        if opt.phase is 'test':
            pass
            # print("===========================================LOADING parameters====================================================")
            # model_parameters = self.load_network(model, 'G', 'best_vanila')
            # model.load_state_dict(model_parameters)
        else:
            self.criterion_l1 = nn.L1Loss()
            self.criterion_adv = GANLoss('hinge').cuda()
            # wgan-gp is somehow not compatible with the spectral normalization (https://github.com/heykeetae/Self-Attention-GAN)
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.dis_optimizer = torch.optim.Adam(self.dis_net.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def save(self, label):
        self.save_network(self.net, self.name+'_g', label)
        if self.opt.isTrain:
            self.save_network(self.dis_net, self.name+'_d', label)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name+'_g', label)
        if self.opt.isTrain:
            self.load_network(self.dis_net, self.name+'_d', label)

    def show_tensorboard_eval(self, num_iter):
        for k, v in self.performance.items():
            self.writer.add_scalar(k, v, num_iter)

    def show_tensorboard(self, num_iter, num_show=4):
        for k, v in self.loss.items():
            self.writer.add_scalar(k, v, num_iter)
        num_show = num_show if self.input.size(0) > num_show else self.input.size(0)

        pred = self.prediction[-1][:num_show] * self.mask[:num_show] + self.input[:num_show]
        self.writer.add_image('prediction', torchvision.utils.make_grid(pred.detach()), num_iter)

        img = self.input[:num_show]
        self.writer.add_image('image', torchvision.utils.make_grid(img), num_iter)

    def set_input(self, image, mask):
        image = image.cuda()
        mask = mask.unsqueeze(1).cuda()
        self.input = image*(1-mask)
        self.targets = image
        self.mask = mask

    def forward(self, it=None):
        # print("We are Forwarding !!")
        if it is not None and it % self.opt.d_repeat == 0:
            self.prediction = self.net.forward(self.input, self.mask)
        else:
            with torch.no_grad():
                self.prediction = self.net.forward(self.input, self.mask)
        if not isinstance(self.prediction, tuple):
            self.prediction = (self.prediction, )

    def test(self, image, mask, name, WW, HH):
        self.set_input(image, mask)
        b, c, h, w = image.shape
        with torch.no_grad():
            self.forward()
            outputs = self.prediction[-1] * 255
            targets = self.targets * 255
            mse = ((outputs-targets)**2*self.mask).sum(3).sum(2).sum(1)
            mse /= (c*h*w)
            psnr = 10 * np.log10(255.0*255.0 / (mse+1e-8))
        return psnr.sum().item()
        # outputs = outputs.detach().cpu().numpy() * 255
        # outputs = outputs.transpose((0, 2, 3, 1))
        # for ii, msk in enumerate(outputs):
        #     msk = Image.fromarray(msk.astype(np.uint8))
        #     msk = msk.resize((WW[ii], HH[ii]))
        #     msk.save('{}/{}.jpg'.format(self.opt.results_dir, name[ii]), 'PNG')

    def random_box(self):
        # [y, x, h, w]
        return [random.randint(0, self.opt.imageSize/2-1), random.randint(0, self.opt.imageSize/2-1),
                int(self.opt.imageSize/2), int(self.opt.imageSize/2)]

    def box_clip(self, image, box):
        return image[:, :, box[0]:box[0]+box[2], box[1]:box[1]+box[3]]


    def backward(self, it):
        box = self.random_box()
        # discriminator loss
        dis_input_real = self.targets
        dis_input_fake = self.prediction[-1].detach() * self.mask + self.targets * (1-self.mask)

        patch_real = self.box_clip(dis_input_real, box)
        patch_fake = self.box_clip(dis_input_fake, box)
        patch_mask = self.box_clip(self.mask, box)

        dis_real, patch_dis_real = self.dis_net(dis_input_real, patch_real, self.mask, patch_mask)  # in: [rgb(3)]
        dis_fake, patch_dis_fake = self.dis_net(dis_input_fake, patch_fake, self.mask, patch_mask)  # in: [rgb(3)]
        dis_real_loss = self.criterion_adv(dis_real, True)
        dis_fake_loss = self.criterion_adv(dis_fake, False)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2 * self.w_adv
        loss = dis_loss
        self.loss['dis'] = dis_loss.item()

        # dis_real_loss = self.criterion_adv(patch_dis_real, True)
        # dis_fake_loss = self.criterion_adv(patch_dis_fake, False)
        # patch_dis_loss = (dis_real_loss + dis_fake_loss) / 2
        # self.loss['patch_dis'] = patch_dis_loss.item()
        # loss += patch_dis_loss
        loss.backward()

        if it % self.opt.d_repeat == 0:

            loss_var = sum([self.criterion_l1(pred, self.targets) for pred in self.prediction]) + \
                       self.criterion_l1(dis_input_fake, self.targets)

            self.loss['l1'] = loss_var.item()
            loss = loss_var

            patch_fake = self.box_clip(self.prediction[-1], box)
            dis_input_fake = self.prediction[-1] * self.mask + self.targets * (1-self.mask)
            gen_fake, patch_gen_fake = self.dis_net(dis_input_fake, patch_fake, self.mask, patch_mask)  # in: [rgb(3)]
            gen_gan_loss = self.criterion_adv(gen_fake, True) * self.w_adv
            self.loss['gen'] = gen_gan_loss.item()
            loss += gen_gan_loss


        # patch_gen_gan_loss = self.criterion_adv(patch_fake, True) * self.w_adv
        # self.loss['patch_gen'] = patch_gen_gan_loss.item()
        # loss += patch_gen_gan_loss

            loss.backward()


    def optimize_parameters(self, it):
        # if it % 10000 == 0:
        #     self.w_adv *= 2
        self.forward(it)
        self.optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        self.backward(it)
        self.optimizer.step()
        self.dis_optimizer.step()


    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

