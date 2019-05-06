from .base_options import _BaseOptions

class TestOptions(_BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.isTrain = False
