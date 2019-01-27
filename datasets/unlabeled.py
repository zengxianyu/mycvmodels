import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
import pdb
import random
from .base_data import _BaseData


class ImageFiles(_BaseData):
    def __init__(self, img_dir, size=256, trining=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(ImageFiles, self).__init__(crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.size = size
        self.training = trining
        names = os.listdir(img_dir)
        self.img_filenames = list(map(lambda x: os.path.join(img_dir, x), names))
        names = list(map(lambda x: '.'.join(x.split('.')[:-1]), names))
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        name = self.names[index]
        img = Image.open(img_file)
        WW, HH = img.size
        if self.crop is not None:
            img, = self.random_crop(img)
        if self.rotate is not None:
            img, = self.random_rotate(img)
        if self.flip:
            img, = self.random_flip(img)
        img = img.resize((self.size, self.size))

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
        if self.training:
            return img
        else:
            return img, name, WW, HH


if __name__ == "__main__":
    sb = ImageFiles('../../data/datasets/ILSVRC14VOC/images')
    pdb.set_trace()
