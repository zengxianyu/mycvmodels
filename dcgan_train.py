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
