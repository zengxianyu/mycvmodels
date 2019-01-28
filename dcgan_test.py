# coding=utf-8

import random
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from datasets import ImageFiles
from models import GANModel

from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

opt.mean = [0.5, 0.5, 0.5]
opt.std = [0.5, 0.5, 0.5]
ngpu = len(opt.gpu_ids)
nz = 100
ngf = 64
ndf = 64

model = GANModel(opt, nz, ngf, ndf)
model.load(99000)


model.switch_to_eval()

for i in tqdm(range(100), desc='testing'):
    model.test(i)


print("We are done")
