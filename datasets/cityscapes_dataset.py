import os

from .base_dataset import BaseDataset
from utils.image_folder import get_img_filepaths


class CityscapesDataset(BaseDataset):
    """Popular car dashcam dataset showing urban scenes: https://www.cityscapes-dataset.com/

    General features:
        - 1024 x 2048 resolution
        - 30 classes
        - Instance-wise annotation (i.e use_instance_maps = True)
        - 5000 finely-annotated images
        - 20 000 coarsely-annotated images (no instance maps)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = super().modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode="fixed")
        parser.set_defaults(resize_to=1024)
        parser.set_defaults(crop_to=1024)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(n_classes=30)
        parser.set_defaults(batch_size=16)
        parser.set_defaults(use_instance_maps=True)

        opt, _ = parser.parse_known_args()
        if hasattr(opt, "num_upsampling_layers"):
            parser.set_defaults(num_upsampling_layers="more")

        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = "train" if opt.is_train else "val"

        image_dir = os.path.join(root, "leftImg8bit", phase)
        image_paths = get_img_filepaths(image_dir, recursively=True)

        label_dir = os.path.join(root, "gtFine", phase)
        label_paths_all = get_img_filepaths(label_dir, recursively=True)
        label_paths = [p for p in label_paths_all if p.endswith("_labelIds.png")]

        if opt.use_instance_maps:
            instance_paths = [p for p in label_paths_all if p.endswith("_instanceIds.png")]
        else:
            instance_paths = []

        return image_paths, label_paths, instance_paths

    def filenames_match(self, path1, path2):
        """Cityscapes overrides this function as it has a specific naming convention."""
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)

        # Compare the first 3 components, [city]_[id1]_[id2]
        return "_".join(name1.split("_")[:3]) == "_".join(name2.split("_")[:3])