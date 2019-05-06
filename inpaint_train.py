# coding=utf-8

import pdb
import time
import torch
import sys
import numpy as np
from tqdm import tqdm
from models import InpaintModel
from datasets import ImageNetMask
import json
import os


from options.train_options import TrainOptions
opt = TrainOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt.parser.set_defaults(name='inpaint')
opt = opt.parse()


home = os.path.expanduser("~")

train_img_dir = '%s/data/datasets/ILSVRC12_image_train'%home
val_img_dir = '%s/data/datasets/ILSVRC12_image_val'%home
val_mask_dir = '%s/gitmodel/mycvmodels/imagenet_mask'%home


train_loader = torch.utils.data.DataLoader(
    ImageNetMask(train_img_dir, size=256, training=True,
                 crop=None, rotate=None, flip=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)
# train_loader.dataset.__getitem__(0)

val_loader = torch.utils.data.DataLoader(
    ImageNetMask(val_img_dir, mask_root=val_mask_dir, size=(680, 512), training=False,
                 crop=None, rotate=None, flip=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    psnr = 0
    batch = 0
    for i, (img, msk, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        batch += img.size(0)
        psnr += model.test(img, msk, name, WW, HH)
    model.switch_to_train()
    # psnr = compute_psnr(opt.results_dir, val_img_dir, val_mask_dir)
    model.performance = {'psnr': psnr / batch}
    return psnr


model = InpaintModel(opt)


def train(model):
    print("============================= TRAIN ============================")
    model.switch_to_train()

    train_iter = iter(train_loader)
    it = 0
    log = {'best': 0, 'best_it': 0}

    for i in tqdm(range(opt.train_iters), desc='train'):
        if it >= len(train_loader):
            train_iter = iter(train_loader)
            it = 0
        image, mask = train_iter.next()
        it += 1

        model.set_input(image, mask)
        model.optimize_parameters()

        if i % opt.display_freq == 0:
            model.show_tensorboard(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            psnr = test(model)
            model.show_tensorboard_eval(i)
            log[i] = {'psnr': psnr}
            # if psnr > log['best']:
            #     log['best'] = psnr
            #     log['best_it'] = i
            #     model.save('best')
            print(u'最大psnr: %.4f, 这次psnr: %.4f'%(log['best'], psnr))
            with open(model.save_dir+'/'+'train-log.json', 'w') as outfile:
                json.dump(log, outfile)


train(model)

print("We are done")
