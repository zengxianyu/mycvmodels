# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import DepthModel
from datasets import ImageFiles
from evaluate_depth import rel_log10_rms
import json
import os


from options.test_options import TestOptions
opt = TestOptions().parse()


home = os.path.expanduser("~")

val_img_dir = '%s/data/datasets/depth_dataset/make3d/Test134'%home
val_gt_dir = '%s/data/datasets/depth_dataset/make3d/depth'%home

model_label = '_best'


val_loader = torch.utils.data.DataLoader(
    ImageFiles(val_img_dir, crop=None, flip=False,
               mean=opt.mean, std=opt.std),
    # Make3d(val_img_dir, val_gt_dir,
    #        crop=0.9, flip=True, rotate=None, size=opt.imageSize,
    #        mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    rel, log10, rms = rel_log10_rms(opt.results_dir, val_gt_dir)
    print(u'rel: %.4f, log10: %.4f, rms: %.4f'%(rel, log10, rms))
    return rel, log10, rms


model = DepthModel(opt)
model.load(model_label)

rel, log10, rms = test(model)
test_log = {'rel': rel, 'log10': log10, 'rms': rms}
with open(model.save_dir+'/'+'test_-{}.json'.format(model_label), 'w') as outfile:
    json.dump(test_log, outfile)

print("We are done")
