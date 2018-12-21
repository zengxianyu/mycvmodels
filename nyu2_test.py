# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import DepthModel
from datasets import NYU2
from evaluate_depth import rel_log10_rms
import json
import os


from options.test_options import TestOptions
opt = TestOptions().parse()


home = os.path.expanduser("~")

img_dir = '%s/data/datasets/depth_dataset/NYU2/data/image'%home
gt_dir = '%s/data/datasets/depth_dataset/NYU2/data/depth'%home
split_file = '%s/data/datasets/depth_dataset/NYU2/data/split.npz'%home

model_label = '_best'

val_loader = torch.utils.data.DataLoader(
    NYU2(img_dir, gt_dir, split_file,
         crop=None, flip=False, rotate=None, size=opt.imageSize,
         mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=False, num_workers=4, pin_memory=True)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    rel, log10, rms = rel_log10_rms(opt.results_dir, gt_dir)
    model.switch_to_train()
    print(u'rel: %.4f, log10: %.4f, rms: %.4f'%(rel, log10, rms))
    return rel, log10, rms


model = DepthModel(opt)
model.load(model_label)
rel, log10, rms = test(model)
test_log = {'rel': rel, 'log10': log10, 'rms': rms}
with open(model.save_dir+'/'+'test_-{}.json'.format(model_label), 'w') as outfile:
    json.dump(test_log, outfile)

print("We are done")
