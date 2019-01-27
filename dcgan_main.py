# from __future__ import print_function
# import argparse
# import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from datasets import ImageFiles
from models import GANModel
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
# import torch.optim as optim
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
# parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
# parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
# parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--manualSeed', type=int, help='manual seed')
#
# opt = parser.parse_args()
# print(opt)
#
# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass
#


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

dataset = ImageFiles(img_dir='/home/zhang/zhs/datasets/syn/bg4000/images', size=opt.imageSize, trining=True,
                 crop=0.9, rotate=None, flip=True, mean=opt.mean, std=opt.std)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=4)
model = GANModel(opt, nz, ngf, ndf)


def train(model):
    print("============================= TRAIN ============================")
    model.switch_to_train()

    train_iter = iter(train_loader)
    it = 0

    for i in tqdm(range(opt.train_iters), desc='train'):
        if it >= len(train_loader):
            train_iter = iter(train_loader)
            it = 0
        img = train_iter.next()
        it += 1

        model.set_input(img)
        model.optimize_parameters_d()
        model.optimize_parameters_g()

        if i % opt.display_freq == 0:
            model.show_tensorboard(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            model.test()


train(model)

print("We are done")
