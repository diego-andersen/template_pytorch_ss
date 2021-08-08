import os
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T

from utils.sorting import natural_sort

class BaseDataset(data.Dataset):
    """
    Base dataset class from which all other datasets are subclassed.
    Performs basic transformations and checks that are common to all image datasets,
    as well as globally-used __len__() and __getitem__() methods.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true', help="Skip sanity check of correct label-image filename pairing.")
        parser.add_argument('--use_instance_maps', action='store_true', help="Use instance maps on top of semantic labels, meaning that each object of the same type is labelled differently.")

        return parser

    def initialize(self, opt):
        self.opt = opt

        image_paths, label_paths, instance_paths = self.get_paths(opt)

        natural_sort(image_paths)
        natural_sort(label_paths)
        if not opt.no_instance_maps:
            natural_sort(instance_paths)

        image_paths = image_paths[:opt.max_dataset_size]
        label_paths = label_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(image_paths, label_paths):
                assert self.filenames_match(path1, path2), \
                    "Label-image pair ({}, {}) do not have matching filenames. Please check your dataset or use --no_pairing_check to bypass this.".format(path1, path2)

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.instance_paths = instance_paths

        self.dataset_size = len(self.image_paths)

    def get_paths(self, opt):
        image_paths = []
        label_paths = []
        instance_paths = []
        assert False, "A subclass of BaseDataset must override self.get_paths()"
        return image_paths, label_paths, instance_paths

    def filenames_match(self, path1, path2):
        """Checks whether two filenames match, ignoring the extension.
        Used to ensure that image-label pairs belong together.
        """

        name1 = os.path.splitext(os.path.basename(path1))[0]
        name2 = os.path.splitext(os.path.basename(path2))[0]
        return name1 == name2

    def postprocess(self, input_dict):
        """Allow subclasses to add dataset-specific postprocessing to images
        at the end of __getitem__().
        """
        return input_dict

    def __getitem__(self, idx):
        # Input image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        params = get_params(self.opt, image.size)

        transform_image = get_transforms(self.opt, params)
        image_tensor = transform_image(image)

        # Label Image
        label_path = self.label_paths[idx]
        label = Image.open(label_path)

        transform_label = get_transforms(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255
        label_tensor[label_tensor == 255] = self.opt.n_classes  # 'Unknown' class label == opt.n_classes
        label_tensor = label_tensor.squeeze(0)
        label_tensor = label_tensor.type(torch.LongTensor)

        # Optionally use instance maps
        if self.opt.use_instance_maps:
            instance_path = self.instance_paths[idx]
            instance = Image.open(instance_path)

            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)
        else:
            instance_tensor = 0

        input_dict = {
            'image': image_tensor,
            'label': label_tensor,
            'instance': instance_tensor,
            'path': image_path
        }

        # Give subclasses a chance to modify the final output
        input_dict = self.postprocess(input_dict)

        return input_dict

    def __len__(self):
        return self.dataset_size


def get_params(opt, size):
    """Calculate random origin point from which to apply cropping, as well
    as whether to flip the image.
    """
    w, h = size

    if "scale height" in opt.preprocess:
        new_h = opt.resize_to
        new_w = opt.resize_to * w // h
    elif "scale width" in opt.preprocess:
        new_w = opt.resize_to
        new_h = opt.resize_to * h // w
    else:
        new_w = w
        new_h = h
    if "crop" in opt.preprocess:
        x = random.randint(0, max(0, new_w - opt.crop_to[0]))
        y = random.randint(0, max(0, new_h - opt.crop_to[1]))
    else:
        x = y = 0

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transforms(opt, params, method=Image.BILINEAR, normalize=True, to_tensor=True):
    """Construct a list of transformations to apply to data samples as they are loaded.

    Parameters:
        opt {BaseOptions subclass}: User options.
        method {PIL.Image.attr}: Indicates how to interpolate pixels during transforms.
        normalize {bool}: Normalise the colour channels to [0.5, 0.5, 0.5].
        to_tensor {bool}: Change the Image into a tensor after transforms are done.

    Returns:
        {torchvision.transforms.Compose}: Ordered list of transforms to be applied to the image.
    """
    transform_list = []

    if 'scale_width' in opt.preprocess:
        transform_list.append(T.Lambda(lambda img: __scale_width(img, opt.resize_to, method)))
    elif 'scale_height' in opt.preprocess:
        transform_list.append(T.Lambda(lambda img: __scale_height(img, opt.resize_to, method)))

    if 'crop' in opt.preprocess:
        transform_list.append(T.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_to)))

    if opt.preprocess == 'none':
        base = 32
        transform_list.append(T.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.is_train and not opt.no_flip:
        transform_list.append(T.Lambda(lambda img: __flip(img, params['flip'])))

    if to_tensor:
        transform_list += [T.ToTensor()]

    if normalize:
        transform_list += [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return T.Compose(transform_list)


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    original_w, original_h = img.size
    h = int(round(original_h / base) * base)
    w = int(round(original_w / base) * base)

    if (h == original_h) and (w == original_w):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_w, method=Image.BICUBIC):
    original_w, original_h = img.size
    if (original_w == target_w):
        return img
    w = target_w
    h = int(target_w * original_h / original_w)
    return img.resize((w, h), method)


def __scale_height(img, target_h, method=Image.BICUBIC):
    original_w, original_h = img.size
    if (original_h == target_h):
        return img
    h = target_h
    w = int(target_h * original_w / original_h)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    x, y = pos
    target_w, target_h = size
    return img.crop((x, y, x + target_w, y + target_h))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
