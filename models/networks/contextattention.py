import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tools import spectral_norm, weight_init
from .inpaint_blocks import *
import pdb


def batch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)

    return out


class ContextAttention(nn.Module):
    def __init__(self, bkg_patch_size=3, smooth_kernel_size=3):
        super(ContextAttention, self).__init__()
        self.bkg_patch_size = bkg_patch_size
        self.smooth_kernel_size = int(smooth_kernel_size) / 2 *2 + 1

    def forward(self, x, mask):
        b, c, h, w = x.shape
        normed_x = x / torch.sqrt((x**2).sum(3, keepdim=True).sum(2, keepdim=True) + 1e-8)
        bkg_kernel = F.unfold(input=normed_x, kernel_size=(self.bkg_patch_size, self.bkg_patch_size), padding=1)
        bkg_kernel = bkg_kernel.transpose(1, 2).view(b, h*w, c, self.bkg_patch_size, self.bkg_patch_size)
        bkg_mask = F.unfold(input=(1-mask), kernel_size=(self.bkg_patch_size, self.bkg_patch_size), padding=1)
        bkg_mask = bkg_mask.transpose(1, 2).mean(2, keepdim=True)[..., None]
        bkg_mask[bkg_mask>0] = 1
        cos_similar = batch_conv2d(normed_x, weight=bkg_kernel, padding=1)
        cos_similar = nn.ReflectionPad2d(padding=(self.smooth_kernel_size-1)/2)(cos_similar)
        cos_similar = F.softmax(cos_similar, dim=1)
        smooth_weight = torch.ones(h*w, 1, self.smooth_kernel_size, self.smooth_kernel_size) / 9.0
        if torch.cuda.is_available():
            smooth_weight = smooth_weight.cuda()
        cos_similar = F.conv2d(cos_similar, weight=smooth_weight, groups=h*w)
        cos_similar = cos_similar * bkg_mask
        cos_similar = cos_similar * mask
        attn_x = torch.bmm(x.view(b, c, -1), cos_similar.view(b, h*w, h*w)).view(b, c, h, w).contiguous()
        return attn_x
        # return cos_similar


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
        self.CAttn = ContextAttention()
        # self.up = Up_Module(out_ch * 8, 3, isRefine=True)
        # self.up = Up_Module(384, 3, isRefine=True)
        self.up = Up_Module(out_ch*8, 3, isRefine=True)

    def forward(self, x, mask):
        # conv branch
        x = torch.cat((x, mask), dim=1)
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)
        resized_mask = F.interpolate(mask, scale_factor=0.25, mode='nearest')

        # attention branch
        attn_x = self.down_attn_branch(x)
        attn_x = self.CAttn(attn_x, mask=resized_mask)  # attn_x => B x 128(32*4) x W/4 x H/4

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

        # global_y = self.global_discriminator(x)
        global_y = self.global_discriminator(torch.cat((x, mask), dim=1))
        # local_y = self.local_discriminator(local_x)
        # return global_y, local_y  # B x 256*(256 or 512)
        return global_y, None


if __name__ == "__main__":
    cattn = ContextAttention().cuda()
    x = torch.Tensor(4, 8, 8, 8).normal_().cuda()
    mask = torch.zeros(4, 1, 8, 8).cuda()
    mask[:, :, :4, :4] = 1
    sb = cattn(x, mask)

