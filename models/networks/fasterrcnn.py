import torch
import torch.nn as nn
from torch.autograd.variable import Variable

# from .densenet import *
# from .resnet import *
# from .vgg import *

from .densenet import *
from .resnet import *
from .vgg import *

from ..tools import *
from .base_network import BaseNetwork

import numpy as np
import sys
thismodule = sys.modules[__name__]
import pdb


class RPN(nn.Module, BaseNetwork):
    def __init__(self, pretrained=True, base='vgg16', c_hidden=512):
        super(RPN, self).__init__()
        if not (base == 'vgg16'):
            raise NotImplementedError
        dims = dim_dict[base][::-1]
        self.input_conv = nn.Conv2d(dims[0], c_hidden, kernel_size=3, padding=1)
        self.output_mask = nn.Conv2d(c_hidden, 18, kernel_size=1)
        self.output_pos = nn.Conv2d(c_hidden, 36, kernel_size=1)
        self.apply(weight_init)

        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature.classifier = None
        self.apply(fraze_bn)


    def forward(self, x):
        feat = self.feature(x)
        hidden = self.input_conv(feat)
        bsize, _, fsize, _ = hidden.size()
        pred_mask = self.output_mask(hidden)
        pred_pos = self.output_pos(hidden)
        return (pred_mask.view(bsize, 2, 9, fsize, fsize), pred_pos.view(bsize, 4, 9, fsize, fsize)), feat


class ROIHead(nn.Module, BaseNetwork):
    def __init__(self, pretrained=True, base='vgg16', n_classes=20):
        super(ROIHead, self).__init__()
        self.fc_cls = nn.Linear(4096, n_classes+1)
        self.fc_loc = nn.Linear(4096, (n_classes+1)*4)
        self.fc_pre = nn.Linear(25088, 4096)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)

        # if 'vgg' in base:
        #     base_net = getattr(thismodule, base)(pretrained=pretrained)
        #     fc_pre = base_net.classifier
        #     self.fc_pre = nn.Sequential(*(list(fc_pre)[:6]))
        #     if not pretrained:
        #         for m in self.fc_pre.modules():
        #             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Linear):
        #                 m.weight.data.normal_(0.0, 0.01)
        #                 m.bias.data.fill_(0)
        # else:
        #     raise NotImplementedError

    def forward(self, feat):
        bsize = feat.size(0)
        feat = self.fc_pre(feat.view(bsize, -1))
        cls = self.fc_cls(feat)
        loc = self.fc_loc(feat)
        return cls, loc.view(bsize, -1, 4)


if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
