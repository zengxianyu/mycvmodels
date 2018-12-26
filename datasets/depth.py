import os
import numpy as np
import PIL.Image as Image
import torch
from scipy import io
from torch.utils import data
import pdb
import random
from base_data import _BaseData


class _BaseDepthData(_BaseData):
    def __init__(self, size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(_BaseDepthData, self).__init__(crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.size = size
        self.training = training

    def __len__(self):
        return len(self.names)

    def _load_depth(self, gt_file):
        gt = Image.open(gt_file)
        return gt

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        gt = self._load_depth(gt_file)
        name = self.names[index]
        WW, HH = gt.size
        img = img.resize((WW, HH))
        if self.crop is not None:
            img, gt = self.random_crop(img, gt)
        if self.rotate is not None:
            img, gt = self.random_rotate(img, gt)
        if self.flip:
            img, gt = self.random_flip(img, gt)
        img = img.resize((self.size, self.size))
        gt = gt.resize((self.size, self.size))

        img = np.array(img, dtype=np.float64) / 255.0
        gt = np.array(gt, dtype=np.float64)
        # if np.sum(gt > 1e-8) > 10:
        #     gt[gt > np.percentile(gt[gt > 1e-8], 98)] = 0
        #     gt[gt < np.percentile(gt[gt > 1e-8], 1)] = 0
        mask = (gt>1e-8).astype(np.float32)
        gt[gt<1e-8] = 1e-8
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
        gt = torch.from_numpy(gt).float()
        mask = torch.from_numpy(mask).float()
        if self.training:
            return img, gt, mask
        else:
            return img, name, WW, HH


class PBR(_BaseDepthData):
    def __init__(self, img_dir, gt_dir, split_file, img_format='jpg', gt_format='png', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None, postfix='color'):
        super(PBR, self).__init__(crop=crop, rotate=rotate, flip=flip, mean=mean, std=std, size=size, training=training)
        folder_names = open(split_file, 'r').readlines()
        folder_names = [n.strip('\n') for n in folder_names]
        names = ['{}/{}'.format(f, ('.'.join(n.split('.')[:-1])).split('_')[0])
                 for f in folder_names for n in os.listdir('{}/{}'.format(img_dir, f))]
        self.img_filenames = [os.path.join(img_dir, name+'_'+postfix+'.'+img_format) for name in names]
        self.gt_filenames = [os.path.join(gt_dir, name+'_depth.'+gt_format) for name in names]
        self.names = [name+'_depth' for name in names]

    def _load_depth(self, gt_file):
        gt = Image.open(gt_file)
        gt = np.array(gt, dtype=np.float64) / 1000.0  # to meter
        gt = Image.fromarray(gt)
        return gt


class NYU2(_BaseDepthData):
    def __init__(self, img_dir, gt_dir, split_file, img_format='png', gt_format='npy', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(NYU2, self).__init__(crop=crop, rotate=rotate, flip=flip, mean=mean, std=std,
                                   size=size, training=training)
        names = np.load(split_file)['train_id'] if training else np.load(split_file)['test_id']
        names = ['%04d'%name for name in names]
        self.img_filenames = [os.path.join(img_dir, name+'.'+img_format) for name in names]
        self.gt_filenames = [os.path.join(gt_dir, name+'.'+gt_format) for name in names]
        self.names = names

    def __len__(self):
        return len(self.names)

    def _load_depth(self, gt_file):
        gt = np.load(gt_file)
        gt = Image.fromarray(gt)
        return gt


class Make3d(_BaseDepthData):
    def __init__(self, img_dir, gt_dir, img_format='jpg', gt_format='mat', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(Make3d, self).__init__(crop=crop, rotate=rotate, flip=flip, mean=mean,
                                     std=std, size=size, training=training)
        names1 = ['-'.join('.'.join(name.split('.')[:-1]).split('-')[1:]) for name in os.listdir(img_dir)]
        names2 = ['-'.join('.'.join(name.split('.')[:-1]).split('-')[1:]) for name in os.listdir(gt_dir)]
        names = list(set(names1)&set(names2))
        self.img_filenames = [os.path.join(img_dir, 'img-'+name+'.'+img_format) for name in names]
        self.gt_filenames = [os.path.join(gt_dir, 'depth_sph_corr-'+name+'.'+gt_format) for name in names]
        self.names = names

    def _load_depth(self, gt_file):
        gt = io.loadmat(gt_file)
        gt = gt['Position3DGrid'][:, :, 3]
        gt = Image.fromarray(gt)
        return gt


if __name__ == "__main__":
    dset = Make3d('/home/zeng/data/datasets/depth_dataset/make3d/Train400Img',
                  '/home/zeng/data/datasets/depth_dataset/make3d/Train400Depth')
    sb = dset.__getitem__(0)
    pdb.set_trace()
