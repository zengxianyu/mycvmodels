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


from options.test_options import TestOptions
opt = TestOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt.parser.set_defaults(name='inpaint_adv_selfa4c')
opt = opt.parse()


home = os.path.expanduser("~")

val_img_dir = '%s/data/datasets/imagenet10val'%home
val_mask_dir = '%s/data/datasets/imagenet10mask'%home


val_loader = torch.utils.data.DataLoader(
    ImageNetMask(val_img_dir, mask_root=val_mask_dir, size=(168, 128), training=False,
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
    psnr /= batch
    model.performance = {'psnr': psnr}
    return psnr


model = InpaintModel(opt)


def test_points(model):
    print("============================= TRAIN ============================")
    model.switch_to_eval()
    names = os.listdir(model.save_dir)
    names = filter(lambda x: '_g.pth' in x, names)
    points = [n.split('_')[1] for n in names]
    if 'best' in points:
        points.remove('best')
    points = [int(p) for p in points]
    points = sorted(points)
    log = {'best': 0}

    for i in tqdm(points, desc='test'):
        model.load(i)
        psnr = test(model)
        model.show_tensorboard_eval(i)
        log[i] = {'psnr': psnr}
        if psnr > log['best']:
            log['best'] = psnr
            log['best_it'] = i
            model.save('best')
        print(u'最大psnr: %d的%.4f, 这次psnr: %.4f'%(i, log['best'], psnr))
        with open(model.save_dir+'/'+'points-log.json', 'w') as outfile:
            json.dump(log, outfile)


test_points(model)

print("We are done")
