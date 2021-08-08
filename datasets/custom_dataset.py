import os

from .base_dataset import BaseDataset
from utils.image_folder import get_img_filepaths

class CustomDataset(BaseDataset):
    """Dataset that loads images from directories
    Use options --image_dir, --label_dir, --instance_dir to specify the directories.
    Images are sorted and paired in alphabetical order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess="scale width, crop")
        parser.set_defaults(resize_to=512)
        parser.set_defaults(crop_to=[256, 256])
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(n_classes=13)
        parser.set_defaults(contains_unknown_label=False)

        parser.add_argument("--image_dir", type=str, required=True,
                            help="Path to the directory that contains photo images")
        parser.add_argument("--label_dir", type=str, required=True,
                            help="Path to the directory that contains label images")
        parser.add_argument("--instance_dir", type=str, default="",
                            help="(Optional) path to the directory that contains instance maps.")
        return parser

    def get_paths(self, opt):
        image_paths = get_img_filepaths(opt.image_dir, recursively=False, read_cache=True)
        label_paths = get_img_filepaths(opt.label_dir, recursively=False, read_cache=True)

        if opt.instance_dir:
            instance_paths = get_img_filepaths(opt.instance_dir, recursively=False, read_cache=True)
        else:
            instance_paths = []

        assert len(image_paths) == len(label_paths), \
            "The number of images in {} and {} do not match.".format(opt.image_dir, opt.label_dir)

        return image_paths, label_paths, instance_paths