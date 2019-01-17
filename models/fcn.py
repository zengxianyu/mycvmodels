import torch
import torch.nn as nn
from torch.autograd.variable import Variable

# from .densenet import *
# from .resnet import *
# from .vgg import *

from densenet import *
from resnet import *
from vgg import *
from funcs import *

import numpy as np
import sys
thismodule = sys.modules[__name__]
import pdb


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition3[-2].register_forward_hook(hook)
    model.features.transition2[-2].register_forward_hook(hook)
    model.classifier = None
    return model


def proc_vgg(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features[3][-2].register_forward_hook(hook)
    model.features[2][-2].register_forward_hook(hook)
    model.classifier = None
    return model


procs = {'densenet169': proc_densenet,
         'vgg16': proc_vgg}


class FCN(nn.Module):
    def __init__(self, pretrained=True, c_output=21, base='densenet169'):
        super(FCN, self).__init__()
        dims = dim_dict[base][::-1]
        self.preds = nn.ModuleList([nn.Conv2d(d, c_output, kernel_size=1) for d in dims])
        if 'vgg' in base:
            self.upscales = nn.ModuleList([
                nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
                nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
                nn.ConvTranspose2d(c_output, c_output, 8, 4, 2),
            ])
        else:
            self.upscales = nn.ModuleList([
                nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
                nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
                nn.ConvTranspose2d(c_output, c_output, 16, 8, 4),
            ])
        self.apply(weight_init)
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        self.apply(fraze_bn)

    def forward(self, x, boxes=None, ids=None):
        self.feature.feats[x.device.index] = []
        x = self.feature(x)
        feats = self.feature.feats[x.device.index]
        feats += [x]
        feats = feats[::-1]
        pred = 0
        for i, feat in enumerate(feats):
            pred = self.preds[i](feat) + pred
            pred = self.upscales[i](pred)
        return pred


if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
