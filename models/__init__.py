from .sal_model import SalModel
from .depth_model import DepthModel
from .seg_model import SegModel
# from .det_model import DetModel
from .gan_model import GANModel
from .inpaint_model import InpaintModel
import importlib

def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    return importlib.import_module(model_name, 'modify_commandline_options')
