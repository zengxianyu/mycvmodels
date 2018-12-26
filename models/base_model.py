import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from datetime import datetime


class _BaseModel:
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.performance = {}

        # tensorboard
        if not os.path.exists(self.save_dir+'/runs'):
            os.mkdir(self.save_dir+'/runs')
        os.system('rm -rf %s/runs/*'%self.save_dir)
        self.writer = SummaryWriter('%s/runs/'%self.save_dir + datetime.now().strftime('%B%d  %H:%M:%S'))

    def set_input(self, input):
        self.input = input

    def show_tensorboard(self, num_iter, num_show=4):
        pass

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self, **kwargs):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self, **kwargs):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def load(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        # print(save_path)
        # model = torch.load(save_path)
        # return model
        network.load_state_dict(torch.load(save_path, map_location={'cuda:%d'%gpu_ids[0]: 'cpu'}))
        # return network

    def update_learning_rate(**kwargs):
        pass
