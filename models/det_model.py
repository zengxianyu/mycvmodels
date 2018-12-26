# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from torch.autograd import Variable
from base_model import _BaseModel
import sys
# from evaluate_sal import fm_and_mae
from tensorboardX import SummaryWriter
from datetime import datetime
from fcn import FCN
from deeplab import DeepLab
from unet import UNet
import ipdb

thismodule = sys.modules[__name__]


def bbox_iou(box, gt_box):
    xmin, ymin, xmax, ymax = box
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
    x_overlap = max(0, min(xmax, gt_xmax) - max(xmin, gt_xmin))
    y_overlap = max(0, min(ymax, gt_ymax) - max(ymin, gt_ymin))
    i_area = x_overlap * y_overlap
    u_area = (xmax - xmin) * (ymax - ymin) + (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) - i_area
    iou = float(i_area) / float(u_area)
    return iou


class RPNLoss(nn.Module):
    def __init__(self, img_size=256, feat_size=16, stride=16):
        super(RPNLoss, self).__init__()
        self.feat_size = 16
        self.scale = 16
        anw, anh = np.meshgrid([1.0, 0.5, 0.25], [1.0, 0.5, 0.25])
        """
        从第1到第9通道对应的xy比例分别是
        [[1.  , 0.5 , 0.25, 1.  , 0.5 , 0.25, 1.  , 0.5 , 0.25],
         [1.  , 1.  , 1.  , 0.5 , 0.5 , 0.5 , 0.25, 0.25, 0.25]]
        """
        anw = anw.ravel() * img_size
        anh = anh.ravel() * img_size
        anw = anw[..., None, None]
        anh = anh[..., None, None]

        an_xct, an_yct = np.meshgrid(range(feat_size), range(feat_size))
        an_xct *= stride
        an_yct *= stride
        an_xct = np.tile(an_xct[None, ...], (9, 1, 1))
        an_yct = np.tile(an_yct[None, ...], (9, 1, 1))
        an_xmin =  an_xct- anw/2
        an_ymin = an_yct - anh/2
        an_xmax = an_xct + anw/2
        an_ymax = an_yct + anh/2

        gt_box = np.array([[0., 0., 120., 90.], [0., 0., 120., 90.]])
        K = 2
        ipdb.set_trace()
        gt_box = gt_box[:, None, None, None, :]
        gt_xmin
        big_x = np.zeros(K, 9, feat_size, feat_size, 4)
        big_x[gt_box>=]



    def bbox2target(self, bboxes):
        pass


    def forward(self, pred, gt, mask):
        pass


if __name__ == "__main__":
    sb = RPNLoss()


#
# class DepthModel(_BaseModel):
#     def __init__(self, opt):
#         _BaseModel.initialize(self, opt)
#         self.name = opt.model + '_' + opt.base
#         self.v_mean = self.Tensor(opt.mean)[None, ..., None, None]
#         self.v_std = self.Tensor(opt.std)[None, ..., None, None]
#         net = getattr(thismodule, opt.model)(pretrained=opt.isTrain and (not opt.from_scratch),
#                                                       c_output=1,
#                                                       base=opt.base)
#         net = torch.nn.parallel.DataParallel(net, device_ids = opt.gpu_ids)
#         self.net = net.cuda()
#         self.input = self.Tensor(opt.batchSize, opt.input_nc,
#                                  opt.imageSize, opt.imageSize)
#         if opt.phase is 'test':
#             pass
#         else:
#             self.criterion = DepthLoss()
#             self.optimizer = torch.optim.Adam(self.net.parameters(),
#                                                 lr=opt.lr)
#
#     def save(self, label):
#         self.save_network(self.net, self.name, label, self.gpu_ids)
#
#     def load(self, label):
#         print('loading %s'%label)
#         self.load_network(self.net, self.name, label, self.gpu_ids)
#
#     def show_tensorboard_eval(self, num_iter):
#         for k, v in self.performance.items():
#             self.writer.add_scalar(k, v, num_iter)
#
#     def show_tensorboard(self, num_iter, num_show=4):
#         self.writer.add_scalar('loss', self.loss, num_iter)
#         num_show = num_show if self.input.size(0) > num_show else self.input.size(0)
#
#         pred_depth = self.prediction[:num_show].detach()
#
#         pred_inv_depth = 1 / pred_depth
#         maximum, _ = pred_inv_depth.max(1)
#         maximum, _ = maximum.max(1)
#         pred_inv_depth /= maximum[:, None, None]
#         pred_inv_depth = pred_inv_depth[:, None, ...]
#         self.writer.add_image('depth', torchvision.utils.make_grid(pred_inv_depth.expand(-1, 3, -1, -1)), num_iter)
#
#         gt = self.targets[:num_show]
#         gt_inv_depth = 1 / gt
#         maximum, _ = gt_inv_depth.max(1)
#         maximum, _ = maximum.max(1)
#         gt_inv_depth /= maximum[:, None, None]
#         gt_inv_depth = gt_inv_depth[:, None, ...]
#         self.writer.add_image('target', torchvision.utils.make_grid(gt_inv_depth.expand(-1, 3, -1, -1)), num_iter)
#         #
#         # maximum, _ = pred.max(1)
#         # maximum, _ = maximum.max(1)
#         # minum, _ = pred.min(1)
#         # minum, _ = minum.min(1)
#         # pred = (pred-minum[:, None, None])/(maximum[:, None, None] - minum[:, None, None])
#         # pred = pred[:, None, ...]
#         # self.writer.add_image('prediction', torchvision.utils.make_grid(1-pred.expand(-1, 3, -1, -1)).detach(), num_iter)
#         #
#         # gt = self.targets[:num_show]
#         # maximum, _ = gt.max(1)
#         # maximum, _ = maximum.max(1)
#         # minum, _ = gt.min(1)
#         # minum, _ = minum.min(1)
#         # gt = (gt-minum[:, None, None])/(maximum[:, None, None] - minum[:, None, None])
#         # gt = gt[:, None, ...]
#         # self.writer.add_image('target', torchvision.utils.make_grid(1-gt.expand(-1, 3, -1, -1)).detach(), num_iter)
#
#         img = self.input[:num_show]*self.v_std + self.v_mean
#         self.writer.add_image('image', torchvision.utils.make_grid(img), num_iter)
#
#     def set_input(self, input, targets=None, mask=None):
#         self.input.resize_(input.size()).copy_(input.cuda())
#         self.targets = targets
#         self.mask = mask
#         if targets is not None:
#             self.targets = self.targets.cuda()
#         if mask is not None:
#             self.mask = self.mask.cuda()
#
#
#     def forward(self):
#         # print("We are Forwarding !!")
#         self.prediction = self.net.forward(self.input)
#         self.prediction = self.prediction.squeeze(1)
#
#
#     def test(self, input, name, WW, HH):
#         self.set_input(input)
#         with torch.no_grad():
#             self.forward()
#         outputs = self.prediction.detach().cpu().numpy()
#
#         maximum = outputs.max(1).max(1)
#         out_img = outputs / maximum[:, None, None]
#         out_img = (out_img*255).astype(np.uint8)
#         for ii, msk in enumerate(out_img):
#             msk = Image.fromarray(msk.astype(np.uint8))
#             msk = msk.resize((WW[ii], HH[ii]))
#             msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')
#             np.save('{}/{}'.format(self.opt.results_dir, name[ii]), outputs[ii])
#
#
#     def backward(self):
#         # Combined loss
#         self.loss_var = self.criterion(self.prediction, self.targets, self.mask)
#         self.loss_var.backward()
#         self.loss = self.loss_var.data[0]
#
#
#     def optimize_parameters(self):
#         self.forward()
#         self.optimizer.zero_grad()
#         self.backward()
#         self.optimizer.step()
#
#
#     def switch_to_train(self):
#         self.net.train()
#
#     def switch_to_eval(self):
#         self.net.eval()

