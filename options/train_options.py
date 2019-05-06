# coding=utf-8
from .base_options import _BaseOptions
import os


class TrainOptions(_BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()

        self.parser.add_argument('--start_it', type=int, default=0, help='recover from saved')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest model')
        self.parser.add_argument('--train_iters', type=int, default=100000, help='training iterations')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='initial learning rate for adam')
        self.parser.add_argument('--beta2', type=float, default=0.9, help='initial learning rate for adam')
        self.parser.add_argument('--from_scratch', action='store_true')

        self.isTrain = True
