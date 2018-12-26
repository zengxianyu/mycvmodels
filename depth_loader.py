import os
import torch
from datasets import PBR, Make3d, ImageFiles, NYU2
from options.train_options import TrainOptions
from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

home = os.path.expanduser("~")


def pbrmlt_train_loader():
    img_dir = '%s/data/datasets/depth_dataset/PBR/images-mlt'%home
    gt_dir = '%s/data/datasets/depth_dataset/PBR/depth'%home
    train_split_file = '%s/data/datasets/depth_dataset/PBR/train.txt'%home
    train_loader = torch.utils.data.DataLoader(
        PBR(img_dir, gt_dir, train_split_file,
            crop=0.9, flip=True, rotate=None, size=opt.imageSize, img_format='png', postfix='mlt',
            mean=opt.mean, std=opt.std, training=True),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def pbrmlt_val_loader():
    img_dir = '%s/data/datasets/depth_dataset/PBR/images-mlt'%home
    gt_dir = '%s/data/datasets/depth_dataset/PBR/depth'%home
    val_split_file = '%s/data/datasets/depth_dataset/PBR/test.txt'%home
    val_loader = torch.utils.data.DataLoader(
        PBR(img_dir, gt_dir, val_split_file,
            crop=None, flip=False, rotate=None, size=opt.imageSize, img_format='png', postfix='mlt',
            mean=opt.mean, std=opt.std, training=False),
        batch_size=opt.batchSize, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader, gt_dir


def pbr_train_loader():
    img_dir = '%s/data/datasets/depth_dataset/PBR/images'%home
    gt_dir = '%s/data/datasets/depth_dataset/PBR/depth'%home
    train_split_file = '%s/data/datasets/depth_dataset/PBR/train.txt'%home
    train_loader = torch.utils.data.DataLoader(
        PBR(img_dir, gt_dir, train_split_file,
            crop=0.9, flip=True, rotate=None, size=opt.imageSize, img_format='jpg', postfix='color',
            mean=opt.mean, std=opt.std, training=True),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def pbr_val_loader():
    img_dir = '%s/data/datasets/depth_dataset/PBR/images'%home
    gt_dir = '%s/data/datasets/depth_dataset/PBR/depth'%home
    val_split_file = '%s/data/datasets/depth_dataset/PBR/test.txt'%home
    val_loader = torch.utils.data.DataLoader(
        PBR(img_dir, gt_dir, val_split_file,
            crop=None, flip=False, rotate=None, size=opt.imageSize, img_format='jpg', postfix='color',
            mean=opt.mean, std=opt.std, training=False),
        batch_size=opt.batchSize, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader, gt_dir


def nyu2_train_loader():
    img_dir = '%s/data/datasets/depth_dataset/NYU2/data/image'%home
    gt_dir = '%s/data/datasets/depth_dataset/NYU2/data/depth'%home
    split_file = '%s/data/datasets/depth_dataset/NYU2/data/split.npz'%home

    train_loader = torch.utils.data.DataLoader(
        NYU2(img_dir, gt_dir, split_file,
             crop=0.9, flip=True, rotate=None, size=opt.imageSize,
             mean=opt.mean, std=opt.std, training=True),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def nyu2_val_loader():
    img_dir = '%s/data/datasets/depth_dataset/NYU2/data/image'%home
    gt_dir = '%s/data/datasets/depth_dataset/NYU2/data/depth'%home
    split_file = '%s/data/datasets/depth_dataset/NYU2/data/split.npz'%home

    val_loader = torch.utils.data.DataLoader(
        NYU2(img_dir, gt_dir, split_file,
             crop=None, flip=False, rotate=None, size=opt.imageSize,
             mean=opt.mean, std=opt.std, training=False),
        batch_size=opt.batchSize, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader, gt_dir


def make3d_train_loader():
    train_img_dir = '%s/data/datasets/depth_dataset/make3d/Train400Img'%home
    train_gt_dir = '%s/data/datasets/depth_dataset/make3d/Train400Depth'%home
    train_loader = torch.utils.data.DataLoader(
        Make3d(train_img_dir, train_gt_dir,
               crop=0.9, flip=True, rotate=None, size=opt.imageSize,
               mean=opt.mean, std=opt.std, training=True),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def make3d_val_loader():
    val_img_dir = '%s/data/datasets/depth_dataset/make3d/Test134'%home
    val_gt_dir = '%s/data/datasets/depth_dataset/make3d/depth'%home
    val_loader = torch.utils.data.DataLoader(
        ImageFiles(val_img_dir, crop=None, flip=False,
                   mean=opt.mean, std=opt.std),
        # Make3d(val_img_dir, val_gt_dir,
        #        crop=0.9, flip=True, rotate=None, size=opt.imageSize,
        #        mean=opt.mean, std=opt.std, training=False),
        batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
    return val_loader, val_gt_dir
