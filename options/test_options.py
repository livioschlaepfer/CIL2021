from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.set_defaults(dataroot="/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/data/val")
        parser.add_argument('--results_dir', type=str, default="/home/svkohler/OneDrive/Desktop/ETH_SS_21/Computational_Intelligence_Lab/Project/Data_GAN/predictions", help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=100000, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        # CRF postprocessing
        parser.add_argument('--crf_post', type=bool, default=True, help='whether to apply crf postprocessing')
        # test transform
        parser.add_argument('--test_trans', type=bool, default=False, help='whether to apply transform at test time and average out')
        # To avoid cropping, the load_size should be the same as crop_size
        #parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
