# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import DepthModel
from datasets import PBR
from evaluate_depth import rel_log10_rms
import json
import os
from depth_loader import *

from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

# val_loader, val_gt_dir = pbr_val_loader()
val_loader, val_gt_dir = nyu2_val_loader()

model_label = '_best'


model = DepthModel(opt)

model.load(model_label)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    rel, log10, rms = rel_log10_rms(opt.results_dir, val_gt_dir)
    model.performance = {'rel': rel, 'log10': log10, 'rms': rms}
    model.switch_to_train()
    print('rel: {}, log10: {}, rms:{}'.format(rel, log10, rms))
    return rel, log10, rms


rel, log10, rms = test(model)

test_log = {'rel': rel, 'log10': log10, 'rms': rms}
with open(model.save_dir+'/'+'nyu2-test_-{}.json'.format(model_label), 'w') as outfile:
    json.dump(test_log, outfile)

print("We are done")
