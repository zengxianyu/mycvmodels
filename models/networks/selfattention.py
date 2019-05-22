import torch
import torch.nn as nn
from ..tools import weight_init
from .inpaint_blocks import *

class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out#, attention


class InpaintGenerator(nn.Module):
    def __init__(self, first_dim=32):
        super(InpaintGenerator, self).__init__()
        self.stage_1 = CoarseNet(4, first_dim)
        self.stage_2 = RefinementNet(4, first_dim)
        self.apply(weight_init)

    def forward(self, masked_img, mask):  # mask : 1 x 1 x H x W
        # stage1
        stage1_output = self.stage_1(masked_img, mask)

        # stage2
        new_masked_img = stage1_output * mask + masked_img.clone() * (1. - mask)
        stage2_output = self.stage_2(new_masked_img, mask)

        return stage1_output, stage2_output



class CoarseNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''

    def __init__(self, in_ch, out_ch):
        super(CoarseNet, self).__init__()
        self.down = Down_Module(in_ch, out_ch)
        self.atrous = Dilation_Module(out_ch * 4, out_ch * 4)
        self.up = Up_Module(out_ch * 4, 3)

    def forward(self, x, mask):
        x = torch.cat((x, mask), dim=1)
        x = self.down(x)
        x = self.atrous(x)
        x = self.up(x)
        return x


class RefinementNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''

    def __init__(self, in_ch, out_ch):
        super(RefinementNet, self).__init__()
        self.down_conv_branch = Down_Module(in_ch, out_ch)
        self.down_attn_branch = Down_Module(in_ch, out_ch, activation=nn.ReLU())
        self.atrous = Dilation_Module(out_ch * 4, out_ch * 4)
        self.SAttn = SelfAttention(out_ch*4, 'relu')
        # self.up = Up_Module(out_ch * 8, 3, isRefine=True)
        self.up = Up_Module(out_ch*8, 3, isRefine=True)

    def forward(self, x, mask):
        # conv branch
        x = torch.cat((x, mask), dim=1)
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)

        # attention branch
        attn_x = self.down_attn_branch(x)
        attn_x = self.SAttn(attn_x)  # attn_x => B x 128(32*4) x W/4 x H/4

        # concat two branches
        deconv_x = torch.cat([conv_x, attn_x], dim=1)  # deconv_x => B x 256 x W/4 x H/4
        x = self.up(deconv_x)

        return x


class InpaintDiscriminator(nn.Module):
    def __init__(self, first_dim=64):
        super(InpaintDiscriminator, self).__init__()
        self.global_discriminator = Flatten_Module(4, first_dim, False)
        # self.local_discriminator = Flatten_Module(3, first_dim, True)
        self.apply(weight_init)


    def forward(self, x, local_x=None, mask=None, local_mask=None):

        global_y = self.global_discriminator(torch.cat((x, mask), dim=1))
        # global_y = self.global_discriminator(x)
        # local_y = self.local_discriminator(local_x)
        # return global_y, local_y  # B x 256*(256 or 512)
        return global_y, None
