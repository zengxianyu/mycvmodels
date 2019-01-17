import torch.nn.functional as F
import torch.nn as nn
import types
import pdb


def conv_b_forward(self, input):
    return F.conv2d(input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups)


def conv_a_forward(self, input):
    weight = self.weight - 1e-4 * self.weight.grad
    bias = self.bias - 1e-4 * self.bias.grad
    return F.conv2d(input, weight=weight, bias=bias, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups)


def convtrans_b_forward(self, input):
    return F.conv_transpose2d(input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)


def convtrans_a_forward(self, input):
    weight = self.weight - 1e-4 * self.weight.grad
    bias = self.bias - 1e-4 * self.bias.grad
    return F.conv_transpose2d(input, weight=weight, bias=bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)


def set_forward_a(m):
    if isinstance(m, nn.Conv2d):
        m.forward = types.MethodType(conv_a_forward, m)
    elif isinstance(m, nn.ConvTranspose2d):
        m.forward = types.MethodType(convtrans_a_forward, m)


def set_forward_b(m):
    if isinstance(m, nn.Conv2d):
        m.forward = types.MethodType(conv_b_forward, m)
    elif isinstance(m, nn.ConvTranspose2d):
        m.forward = types.MethodType(convtrans_b_forward, m)
