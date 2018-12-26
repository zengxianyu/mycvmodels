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


class RPN(nn.Module):
    def __init__(self, pretrained=True, base='densenet169'):
        super(RPN, self).__init__()
        # self.upscales = nn.ModuleList([nn.ConvTranspose2d(ic, oc, 2, 2)
        #                                for ic, oc in zip(dims[:-1], dims[1:])])
        # self.reduce_convs = nn.ModuleList([nn.Conv2d(2*oc, oc, 3, 1, 1)
        #                                    for oc in dims[1:]])
        # self.output_convs = nn.Sequential(nn.Conv2d(dims[-1], c_output, 1, 1),
        #                                   nn.ConvTranspose2d(c_output, c_output, 4, 2, 1))
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
        feats = self.feature.feats[x.device.index]
        feats = feats[::-1]

        for i, feat in enumerate(feats):
            x = self.upscales[i](x)
            x = torch.cat((feats[i], x), 1)
            x = self.reduce_convs[i](x)
        pred = self.output_convs(x)
        return pred


if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
