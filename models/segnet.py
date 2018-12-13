import torch.nn as nn
import torch.nn.functional as F
import torchvision
from functools import reduce
import pdb


class Img2Vec(nn.Module):
    def __init__(self, pretrained=True, c_input=3):
        super(Img2Vec, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                # conv1
                nn.Conv2d(c_input, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True),  # 1/2
            ),
            nn.Sequential(
                # conv2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True),  # 1/4
            ),
            nn.Sequential(
                # conv3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True),  # 1/8
            ),
            nn.Sequential(
                # conv4
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True),  # 1/16
            ),
            nn.Sequential(
                # conv5 features
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True),  # 1/8
            )])
        if pretrained:
            vgg16 = torchvision.models.vgg16_bn(pretrained=pretrained)
            L_vgg16 = list(vgg16.features)
            L_self = reduce(lambda x,y:list(x)+list(y), self.convs)
            L_self[0].weight.data[:, :3] = L_vgg16[0].weight.data
            L_self[0].bias.data = L_vgg16[0].bias.data
            for l1, l2 in zip(L_vgg16[1:-1], L_self[1:]):
                if (isinstance(l1, nn.Conv2d) or
                        isinstance(l1, nn.BatchNorm2d)):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

    def forward(self, x):
        inds = []
        for f in self.convs:
            x, ind = f(x)
            inds += [ind]
        return x, inds[::-1]


class Vec2Img(nn.Module):
    def __init__(self, c_output=21, c_input=512):
        super(Vec2Img, self).__init__()
        self.pred = nn.Conv2d(512, c_output, kernel_size=1)
        self.convs = nn.ModuleList([
            nn.Sequential(
                # conv5 features
                nn.Conv2d(c_input, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            nn.Sequential(
                # conv4
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                # conv3
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                # conv2
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                # conv1
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, c_output, 3, padding=1)
            )
            ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.orthogonal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x, inds):
        x = F.max_unpool2d(x, inds[0], 2, 2)
        x = self.pred(x)
        x = F.upsample(x, scale_factor=16)
        # x = self.convs[0](x)
        # for f, ind in zip(self.convs[1:], inds[1:]):
        #     x = F.max_unpool2d(x, ind, 2, 2)
        #     x = f(x)
        return x


class SegNet(nn.Module):
    def __init__(self, pretrain=False, c_input=3, c_output=21):
        super(SegNet, self).__init__()
        self.img2vec = Img2Vec(pretrained=pretrain, c_input=c_input)
        self.ve22img = Vec2Img(c_output=c_output, c_input=512)

    def forward(self, x):
        x = self.img2vec(x)
        return self.ve22img(*x)