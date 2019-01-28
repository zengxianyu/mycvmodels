# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import DepthModel
from evaluate_depth import rel_log10_rms
import json
import os
from depth_loader import *


from options.train_options import TrainOptions
opt = TrainOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt.parser.set_defaults(name='depth')
opt = opt.parse()


# train_loader = pbr_train_loader()
# val_loader, val_gt_dir = pbr_val_loader()
if 'mlt' in opt.name:
    print('train on pbr-mlt dataset')
    train_loader = pbrmlt_train_loader(opt)
    val_loader, val_gt_dir = pbrmlt_val_loader(opt)
elif 'pbr' in opt.name:
    print('train on pbr-opengl dataset')
    train_loader = pbr_train_loader(opt)
    val_loader, val_gt_dir = pbr_val_loader(opt)
elif 'nyu' in opt.name:
    train_loader = nyu2_train_loader(opt)
    val_loader, val_gt_dir = nyu2_val_loader(opt)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    rel, log10, rms = rel_log10_rms(opt.results_dir, val_gt_dir)
    model.performance = {'rel': rel, 'log10': log10, 'rms': rms}
    model.switch_to_train()
    return rel, log10, rms


def train(model):
    print("============================= TRAIN ============================")

    model.switch_to_train()

    train_iter = iter(train_loader)
    it = 0
    log = {'best': 1000, 'best_it': 0}

    if opt.start_it > 0:
        model.load('_'+str(opt.start_it))
        with open(model.save_dir+'/'+'train-log.json', 'r') as f:
            log = json.load(f)

    for i in tqdm(range(opt.start_it, opt.train_iters), desc='train'):
        # landscape
        if it >= len(train_loader):
            train_iter = iter(train_loader)
            it = 0
        img, gt, mask = train_iter.next()
        it += 1

        model.set_input(img, gt, mask)
        model.optimize_parameters()

        if i % opt.display_freq == 0:
            model.show_tensorboard(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            rel, log10, rms = test(model)
            model.show_tensorboard_eval(i)
            log[it] = {'rel': rel, 'log10': log10, 'rms': rms}
            if rel < log['best']:
                log['best'] = rel
                log['best_it'] = i
                model.save('best')
            print(u'最大rel: it%d的%.4f, 这次it: %d rel: %.4f, log10: %.4f, rms: %.4f'
                  %(log['best_it'], log['best'], i, rel, log10, rms))
            with open(model.save_dir+'/'+'train-log.json', 'w') as outfile:
                json.dump(log, outfile)


model = DepthModel(opt)

train(model)

print("We are done")
