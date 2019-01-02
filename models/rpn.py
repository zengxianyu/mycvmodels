import torch
import torch.nn as nn
from torch.autograd.variable import Variable

# from .densenet import *
# from .resnet import *
# from .vgg import *

from densenet import *
from resnet import *
from vgg import *

import numpy as np
import sys
thismodule = sys.modules[__name__]
import pdb

dim_dict = {
    'vgg16': [64, 128, 256, 512, 512],
}


class RPN(nn.Module):
    def __init__(self, pretrained=True, base='vgg16', c_hidden=512):
        super(RPN, self).__init__()
        dims = dim_dict[base][::-1]
        self.input_conv = nn.Conv2d(dims[0], c_hidden, kernel_size=3, padding=1)
        self.output_mask = nn.Conv2d(c_hidden, 18, kernel_size=1)
        self.output_pos = nn.Conv2d(c_hidden, 36, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)

        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature.feats = {}

        for m in self.feature.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad=False

    def forward(self, x, boxes=None, ids=None):
        self.feature.feats[x.device.index] = []
        x = self.feature(x)
        x = self.input_conv(x)
        bsize, _, fsize, _ = x.size()
        pred_mask = self.output_mask(x)
        pred_pos = self.output_pos(x)
        return pred_mask.view(bsize, 2, 9, fsize, fsize), pred_pos.view(bsize, 4, 9, fsize, fsize)


if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
