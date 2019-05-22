import os
import numpy as np
import PIL.Image as Image
import torch
import pdb
from .base_data import _BaseData
from .create_mask import rectangle_mask, stroke_mask
import random
import math


class ImageNetMask(_BaseData):
    def __init__(self, root, mask_root=None, size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(ImageNetMask, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.root = root
        self.training = training
        if training:
            folders = os.listdir(root)
            self.img_filenames = [os.path.join(root, os.path.join(f, n))
                     for f in folders for n in os.listdir(os.path.join(root, f))]
        else:
            self.mask_root = mask_root
            names = ['.'.join(n.split('.')[:-1]) for n in os.listdir(root)]
            self.names = names
            self.img_filenames = [os.path.join(root, name+'.JPEG') for name in names]
            self.mask_filenames = [os.path.join(mask_root, name+'.png') for name in names]

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file).convert('RGB')
        w, h = img.size
        WW, HH = w, h
        th, tw = self.size
        rate = float(max(th, tw)) / min(h, w)
        img = img.resize((int(math.ceil(w*rate)), int(math.ceil(h*rate))))
        # if w < tw:
        #     img = img.resize((tw, int(float(tw)/w * h)))
        # w, h = img.size
        # if h < th:
        #     img = img.resize((int(float(th)/h * w), th))
        if self.crop is not None:
            img, = self.random_crop(img)
        if self.rotate is not None:
            img, = self.random_rotate(img)
        if self.training:
            img, = self.patch_crop(img)
            mask = stroke_mask(self.size[0], self.size[1]) if random.randint(0, 1) \
                else rectangle_mask(self.size[0], self.size[1], max_hole_size=self.size[0]/2, min_hole_size=self.size[0]/4)
        else:
            mask = Image.open(self.mask_filenames[index])
            img = img.resize(self.size)
            mask = mask.resize(self.size)
            mask = np.array(mask)
            mask[mask != 0] = 1
            name = self.names[index]
        if self.flip:
            img, = self.random_flip(img)
        img = np.array(img, dtype=np.float64) / 255.0
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        if self.training:
            return img, mask
        else:
            return img, mask, name, WW, HH


if __name__ == "__main__":
    sb = ImageNetMask('/home/zeng/data/datasets/ILSVRC12_image_train')
    bb = sb.__getitem__(0)
    pdb.set_trace()
