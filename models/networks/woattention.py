import torch
import torch.nn as nn
from ..tools import weight_init
from .inpaint_blocks import *

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
        # self.up = Up_Module(out_ch * 8, 3, isRefine=True)
        self.up = Up_Module(out_ch*8, 3, isRefine=True)

    def forward(self, x, mask):
        # conv branch
        x = torch.cat((x, mask), dim=1)
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)

        # attention branch
        attn_x = self.down_attn_branch(x)

        # concat two branches
        deconv_x = torch.cat([conv_x, attn_x], dim=1)  # deconv_x => B x 256 x W/4 x H/4
        x = self.up(deconv_x)

        return x


class InpaintDiscriminator(nn.Module):
    def __init__(self, first_dim=64):
        super(InpaintDiscriminator, self).__init__()
        self.global_discriminator = Flatten_Module(4, first_dim, False)
        self.apply(weight_init)


    def forward(self, x, local_x=None, mask=None, local_mask=None):

        global_y = self.global_discriminator(torch.cat((x, mask), dim=1))
        # return global_y, local_y  # B x 256*(256 or 512)
        return global_y, None
