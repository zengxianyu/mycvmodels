import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tools import spectral_norm, weight_init
import pdb


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, K=3, S=1, P=1, D=1, activation=nn.ELU(), use_spectral_norm=False):
        super(Conv, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D, bias= not use_spectral_norm),
                    use_spectral_norm),
                activation
            )
        else:
            self.conv = nn.Sequential(
                spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D, bias=not use_spectral_norm),
                    use_spectral_norm)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


# conv 1~6
class Down_Module(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.ELU()):
        super(Down_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5))

        curr_dim = out_ch
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim * 2, K=3, S=2))
            layers.append(Conv(curr_dim * 2, curr_dim * 2))
            curr_dim *= 2

        layers.append(Conv(curr_dim, curr_dim, activation=activation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


# conv 7~10
class Dilation_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilation_Module, self).__init__()
        layers = []
        dilation = 1
        for i in range(4):
            dilation *= 2
            layers.append(Conv(in_ch, out_ch, D=dilation, P=dilation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


# conv 11~17
class Up_Module(nn.Module):
    def __init__(self, in_ch, out_ch, isRefine=False):
        super(Up_Module, self).__init__()
        layers = []
        curr_dim = in_ch
        if isRefine:
            layers.append(Conv(curr_dim, curr_dim // 2))
            curr_dim //= 2
        else:
            layers.append(Conv(curr_dim, curr_dim))

        # conv 12~15
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(Conv(curr_dim, curr_dim // 2))
            curr_dim //= 2

        layers.append(Conv(curr_dim, curr_dim // 2))
        layers.append(Conv(curr_dim // 2, out_ch, activation=0))

        self.out = nn.Sequential(*layers)

    def forward(self, x):
        output = self.out(x)
        # return (torch.clamp(output, min=-1., max=1.) + 1) / 2
        return (F.tanh(output) + 1)/2


class Flatten_Module(nn.Module):
    def __init__(self, in_ch, out_ch, isLocal=True, use_spectral_norm=True):
        super(Flatten_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5, S=2, P=2, activation=nn.LeakyReLU(), use_spectral_norm=use_spectral_norm))
        curr_dim = out_ch

        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim * 2, K=5, S=2, P=2, activation=nn.LeakyReLU(), use_spectral_norm=use_spectral_norm))
            curr_dim *= 2

        if isLocal:
            layers.append(Conv(curr_dim, curr_dim * 2, K=5, S=2, P=2, activation=nn.LeakyReLU(), use_spectral_norm=use_spectral_norm))
        else:
            layers.append(Conv(curr_dim, curr_dim, K=5, S=2, P=2, activation=nn.LeakyReLU(), use_spectral_norm = use_spectral_norm))

        self.out = nn.Sequential(*layers)
        # self.last = Conv(curr_dim, curr_dim, K=1, S=1, P=0, activation=0)

    def forward(self, x):
        x = self.out(x)
        # x = self.last(x)
        return x  # 2B x 256*(256 or 512); front 256:16*16


if __name__ == "__main__":
    cattn = ContextAttention().cuda()
    x = torch.Tensor(4, 8, 8, 8).normal_().cuda()
    mask = torch.zeros(4, 1, 8, 8).cuda()
    mask[:, :, :4, :4] = 1
    sb = cattn(x, mask)

