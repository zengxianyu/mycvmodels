import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
import pdb


class CropResize(nn.Module):
    def __init__(self, oh, ow, scale=1.0):
        super(CropResize, self).__init__()
        self.oh, self.ow, self.scale = oh, ow, scale

    def forward(self, data, boxes):
        """
        crop and resize rectangle region from image
        :param data: N*C*H*W
        :param boxes: K*4 float tensor, xmin, ymin, xmax, ymax
        :param ids: K dimensional int tensor
        :param oh, ow: height and width of the output tensor
        :return: K*C*H*W
        """
        boxes = boxes / self.scale
        nbox = len(boxes)
        ext_data = data.expand(nbox, -1, -1, -1)
        # ext_data = data[ids]
        _, _, H, W = data.size()
        ys, xs = torch.meshgrid([torch.arange(self.oh), torch.arange(self.ow)])
        xs = xs.cuda()
        ys = ys.cuda()
        ys = ys.unsqueeze(0).expand(nbox, -1, -1)
        xs = xs.unsqueeze(0).expand(nbox, -1, -1)
        a11 = ((boxes[:, 2] - boxes[:, 0]) / self.ow).unsqueeze(1).unsqueeze(2)
        a22 = ((boxes[:, 3] - boxes[:, 1]) / self.oh).unsqueeze(1).unsqueeze(2)
        xmin = boxes[:, 0].unsqueeze(1).unsqueeze(2)
        ymin = boxes[:, 1].unsqueeze(1).unsqueeze(2)
        xi, yi = a11 * xs.float() + xmin, a22 * ys.float() + ymin
        xi = (xi - W / 2) / W * 2
        yi = (yi - H / 2) / H * 2
        grid = torch.stack((xi, yi), 3)
        sb_data = F.grid_sample(ext_data, grid.detach())
        return sb_data


if __name__ == "__main__":
    loader = torch.utils.data.DataLoader(
        SynFolder('/home/zeng/data/datasets/syn_seg'),
        collate_fn=my_collate,
        batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    data_iter = iter(loader)
    data, lbl, ids, boxes, boxes_lbl, names = data_iter.next()
    pdb.set_trace()
    data = F.max_pool2d(data, kernel_size=2, stride=2)
    crop_resize = CropResize(128, 128, 2.0)
    sb_data = crop_resize(data.cuda(), boxes, ids)
    sb_data = sb_data.numpy().transpose((0, 2, 3, 1))
    for sd in sb_data:
        plt.imshow(sd)
        plt.show()
        pdb.set_trace()
    pdb.set_trace()
