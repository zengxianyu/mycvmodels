# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import SegModel
from datasets import VOC
from evaluate_seg import evaluate_iou
import json
import os


from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

home = os.path.expanduser("~")

train_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
train_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClassAug'%home

val_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
val_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClass'%home

train_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/argtrain.txt'%home
val_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'%home

c_output = 21


train_loader = torch.utils.data.DataLoader(
    VOC(train_img_dir, train_gt_dir, train_split,
           crop=0.9, flip=True, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    VOC(val_img_dir, val_gt_dir, val_split,
           crop=0.9, flip=True, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    model.switch_to_train()
    miou = evaluate_iou(opt.results_dir, val_gt_dir, c_output)
    model.performance = {'miou': miou}
    return miou


model = SegModel(opt, c_output, train_loader.dataset.ignored_idx)


def train(model):
    print("============================= TRAIN ============================")
    model.switch_to_train()

    train_iter = iter(train_loader)
    it = 0
    log = {'best': 0, 'best_it': 0}

    for i in tqdm(range(opt.train_iters), desc='train'):
        # landscape
        if it >= len(train_loader):
            train_iter = iter(train_loader)
            it = 0
        img, gt = train_iter.next()
        it += 1

        model.set_input(img, gt)
        model.optimize_parameters()

        if i % opt.display_freq == 0:
            model.show_tensorboard(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            miou = test(model)
            log[i] = {'miou': miou}
            if miou > log['best']:
                log['best'] = miou
                log['best_it'] = i
                model.save('best')
            print(u'最大miou: %.4f, 这次miou: %.4f'%(log['best'], miou))
            with open(model.save_dir+'/'+'train-log.json', 'w') as outfile:
                json.dump(log, outfile)


train(model)

print("We are done")
