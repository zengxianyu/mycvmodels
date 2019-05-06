import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from datetime import datetime
import pdb


class BaseModel:
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.performance = {}
        self.gpu_ids = opt.gpu_ids

        # tensorboard
        if not os.path.exists(self.save_dir+'/runs'):
            os.mkdir(self.save_dir+'/runs')
        os.system('rm -rf %s/runs/*'%self.save_dir)
        self.writer = SummaryWriter('%s/runs/'%self.save_dir + datetime.now().strftime('%Y%m%d_%H:%M:%S'))

    # def set_input(self, input):
    #     self.input = input

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
    def save_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        device = next(network.parameters()).device
        torch.save(network.cpu().state_dict(), save_path)
        network.to(device)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        device = next(network.parameters()).device
        network.load_state_dict(torch.load(save_path, map_location={'cuda:%d' % device.index: 'cpu'}))

    def update_learning_rate(**kwargs):
        pass
