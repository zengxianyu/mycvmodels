from .dcgan import Discriminator, Generator
from .deeplab import DeepLab
from .fasterrcnn import ROIHead, RPN
from .fcn import FCN
from .segnet import SegNet
from .unet import UNet
from . import contextattention
from . import selfattention
from . import woattention

import importlib


def get_option_setter(network_name):
    """Return the static method <modify_commandline_options> of the model class."""
    return importlib.import_module(network_name, 'modify_commandline_options')
