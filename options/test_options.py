from base_options import _BaseOptions

class TestOptions(_BaseOptions):
    def initialize(self):
        _BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.isTrain = False
