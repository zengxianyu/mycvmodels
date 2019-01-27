import torch
import torch.nn as nn
from torch.autograd.variable import Variable

# from .densenet import *
# from .resnet import *
# from .vgg import *

from .densenet import *
from .resnet import *
from .vgg import *

import numpy as np
import sys
from .funcs import *
thismodule = sys.modules[__name__]
import pdb


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition3[-2].register_forward_hook(hook)
    model.features.transition2[-2].register_forward_hook(hook)
    model.features.transition1[-2].register_forward_hook(hook)
    model.features.block0[-2].register_forward_hook(hook)
    model.classifier = None
    return model


def proc_vgg(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features[3][-2].register_forward_hook(hook)
    model.features[2][-2].register_forward_hook(hook)
    model.features[1][-2].register_forward_hook(hook)
    model.features[0][-2].register_forward_hook(hook)
    model.classifier = None
    return model


def proc_resnet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.relu.register_forward_hook(hook)
    model.layer1.register_forward_hook(hook)
    model.layer2.register_forward_hook(hook)
    model.layer3.register_forward_hook(hook)
    model.fc = None
    return model


procs = {'densenet169': proc_densenet,
         'vgg16': proc_vgg,
         'resnet101': proc_resnet}


class UNet(nn.Module):
    def __init__(self, pretrained=True, c_output=21, base='densenet169'):
        super(UNet, self).__init__()
        dims = dim_dict[base][::-1]
        self.upscales = nn.ModuleList([nn.ConvTranspose2d(ic, oc, 2, 2)
                                       for ic, oc in zip(dims[:-1], dims[1:])])
        self.reduce_convs = nn.ModuleList([nn.Conv2d(2*oc, oc, 3, 1, 1)
                                           for oc in dims[1:]])
        if 'vgg' in base:
            self.output_convs = nn.Conv2d(dims[-1], c_output, 1, 1)
        else:
            self.output_convs = nn.Sequential(nn.Conv2d(dims[-1], c_output, 1, 1),
                                              nn.ConvTranspose2d(c_output, c_output, 4, 2, 1))
        self.apply(weight_init)

        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)

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
