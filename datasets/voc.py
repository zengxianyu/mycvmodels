import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
from torchvision import transforms
import pdb
import random
import sys
import matplotlib.pyplot as plt

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


index2color = labelcolormap(21)
index2color = [list(hh) for hh in index2color]
index2name = ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
              'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')

def rotated_rect_with_max_area(w, h, angle):
    """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


class BaseVOC(data.Dataset):
    def __init__(self, root, split='train', crop=None, flip=False, rotate=None,
                 mean=None, std=None):
        super(BaseVOC, self).__init__()
        self.mean, self.std = mean, std
        self.split = split
        self.flip = flip
        self.rotate = rotate
        self.crop = crop
        if split == 'argtrain':
            gt_dir = os.path.join(root, 'SegmentationClassAug')
        else:
            gt_dir = os.path.join(root, 'SegmentationClass')
        img_dir = os.path.join(root, 'JPEGImages')
        names = open('{}/ImageSets/Segmentation/{}.txt'.format(root, split)).read().split('\n')[:-1]
        gt_filenames = ['{}/{}.png'.format(gt_dir, name) for name in names]
        img_filenames = ['{}/{}.jpg'.format(img_dir, name) for name in names]
        self.img_filenames = img_filenames
        self.gt_filenames = gt_filenames
        self.names = names

    def __len__(self):
        return len(self.names)

    def random_crop(self, *images):
        images = list(images)
        sz = [img.size for img in images]
        sz = set(sz)
        assert(len(sz)==1)
        w, h = sz.pop()
        th, tw = int(self.crop*h), int(self.crop*w)
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        results = [img.crop((j, i, j + tw, i + th)) for img in images]
        return tuple(results)

    def random_flip(self, *images):
        if self.flip and random.randint(0, 1):
            images = list(images)
            results = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            return tuple(results)
        else:
            return images

    def random_rotate(self, *images):
        images = list(images)
        sz = [img.size for img in images]
        sz = set(sz)
        assert(len(sz)==1)
        w, h = sz.pop()
        degree = random.randint(-1*self.rotate, self.rotate)
        images_r = [img.rotate(degree, expand=1) for img in images]
        w_b, h_b = images_r[0].size
        w_r, h_r = rotated_rect_with_max_area(w, h, np.radians(degree))
        ws = (w_b - w_r) / 2
        ws = max(ws, 0)
        hs = (h_b - h_r) / 2
        hs = max(hs, 0)
        we = ws + w_r
        he = hs + h_r
        we = min(we, w_b)
        he = min(he, h_b)
        results = [img.crop((ws, hs, we, he)) for img in images_r]
        return tuple(results)


class VOC(BaseVOC):
    def __init__(self, root, split='train', size=256, crop=None, flip=False, rotate=None,
                 mean=None, std=None):
        super(VOC, self).__init__(root=root, split=split, crop=crop, flip=flip, rotate=rotate, mean=mean, std=std)
        self.size = size

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        name = self.names[index]
        if self.split != 'test':
            gt = Image.open(gt_file).convert("P")
            w, h = gt.size
        else:
            w, h = img.size
            gt = np.zeros((h, w), dtype=np.uint8)
            gt = Image.fromarray(gt)
        if self.crop is not None:
            img, gt = self.random_crop(img, gt)
        if self.rotate is not None:
            img, gt = self.random_rotate(img, gt)
        if self.flip:
            img, gt = self.random_flip(img, gt)
        img = img.resize((self.size, self.size))
        gt = gt.resize((self.size, self.size))

        img = np.array(img, dtype=np.float64) / 255.0
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        gt = np.array(gt, dtype=np.int64)
        gt[gt==255] = -1
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt)
        return img, gt, name, w, h


if __name__ == "__main__":
    sb = WSVOCSemi('/home/zeng/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012',
               '/home/zeng/WSL/size256', split='train', img_root='/home/zeng/data/datasets/ILSVRC14VOC/images',
                   msk_root='/home/zeng/WSLfiles/STsoftmax_densenet169/semi_results',
            source_transform=transforms.Compose([transforms.Resize((256, 256))]),
            target_transform=transforms.Compose([transforms.Resize((256, 256))]))
    img, gt, syn, name = sb.__getitem__(0)
    pdb.set_trace()
