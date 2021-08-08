import os

from PIL import Image

from .base_dataset import BaseDataset
from utils.image_folder import get_img_filepaths

class BinaryDataset(BaseDataset):
    """Dataset that loads images from directories
    Use options --image_dir, --label_dir, --instance_dir to specify the directories.
    Images are sorted and paired in alphabetical order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess="none")
        parser.set_defaults(n_classes=2)
        parser.set_defaults(contains_unknown_label=False)

        parser.add_argument("--image_dir", type=str, required=True,
                            help="Path to the directory that contains photo images")
        parser.add_argument("--label_dir", type=str, required=True,
                            help="Path to the directory that contains label images")
        return parser

    def get_paths(self, opt):
        image_paths = get_img_filepaths(opt.image_dir, recursively=False, read_cache=True)
        label_paths = get_img_filepaths(opt.label_dir, recursively=False, read_cache=True)

        instance_paths = []

        assert len(image_paths) == len(label_paths), \
            "The number of images in {} and {} do not match.".format(opt.image_dir, opt.label_dir)

        return image_paths, label_paths, instance_paths

    def __getitem__(self, idx):
        # Input image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        params = super().get_params(self.opt, image.size)

        transform_image = get_transforms(self.opt, params)
        image_tensor = transform_image(image)

        # Label Image
        label_path = self.label_paths[idx]
        label = Image.open(label_path).convert('L')

        transform_label = get_transforms(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        label_tensor[label_tensor == 255] = self.opt.n_classes  # 'Unknown' class label == opt.n_classes

        instance_tensor = None

        input_dict = {
            'image': image_tensor,
            'label': label_tensor,
            'instance': instance_tensor,
            'path': image_path
        }

        # Give subclasses a chance to modify the final output
        input_dict = self.postprocess(input_dict)

        return input_dict