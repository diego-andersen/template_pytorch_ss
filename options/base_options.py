import os
import sys
import argparse
import torch
import pickle

import models
import datasets
import utils.utils as utils


class BaseOptions():
    """Global options that don"t depend on whether the model is in train or test mode.
    Parent class for other option classes, never instantiated directly.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Experiment specifics
        parser.add_argument("--load_opt_from_file", action="store_true", help="When continuing from a saved checkpoint, load the options that were used when creating it.")
        parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Experiments are saved here. Directories are created inside here using --name.")
        parser.add_argument("--name", type=str, default="test_run", help="Name of the experiment. Used to name the top-level directory that outputs are saved in for a particular run.")
        parser.add_argument("--model", type=str, default="test", choices=["test", "deeplabv3"], help="Which model to use.")
        parser.add_argument("--rng_seed", type=int, default=42, help="RNG seed, affects things like dataloader randomness.")
        parser.add_argument("--batch_size", type=int, default=1, help="Input data batch size.")

        # Image I/O
        parser.add_argument("--shuffle", action="store_true", help="Randomise sample order each epoch.")
        parser.add_argument("--preprocess", type=str, default="none", help="Scaling/cropping of images at load time. String is parsed for all keywords in CHOICES.", choices=("scale width", "scale height", "crop", "none"))
        parser.add_argument("--resize_to", type=int, help="Scale images to this size. Used if any of the scale options is selected in --preprocess.")
        parser.add_argument("--crop_to", type=int, nargs=2, help="Crop images to this size. Requires 2 arguments (width, height). Use if crop is selected in --preprocess.")
        parser.add_argument("--no_flip", action="store_true", help="Refrain from randomnly flipping images horizontally while iterating through a training dataset. Does not apply to test/prediction phases.")

        # Dataset
        parser.add_argument("--dataroot", type=str, default="data", help="Root directory from which to load training/test data.")
        parser.add_argument("--dataset_type", type=str, default="custom", help="Archetype of dataset used, e.g. ImageNet, Cityscapes, etc.")
        parser.add_argument("--cache_filelist_write", action="store_true", help="Save the current filelist into a text file, so that it can be loaded faster in the future.")
        parser.add_argument("--cache_filelist_read", action="store_true", help="Read from the file list cache.")
        parser.add_argument("--max_dataset_size", type=int, default=sys.maxsize, help="Maximum number of dataset samples allowed. If the dataset directory contains more than max_dataset_size, only a subset is loaded.")
        parser.add_argument("--n_classes", type=int, help="Number of labelled classes, excluding UNKNOWN class. If UNKNOWN is included, also use --contains_unknown_label.")
        parser.add_argument("--contains_unknown_label", action="store_true", help="Dataset label maps already contain pixels labelled as UNKNOWN (pixel value = 255).")
        parser.add_argument("--no_instance_maps", action="store_true", help="Dataset doesn't have instance maps, i.e. multiple appearances of the same class are labelled the same.")

        # Hardware-related settings
        parser.add_argument("--n_threads", default=0, type=int, help="Number of CPU threads for loading data")
        parser.add_argument("--reserve_one_core", action="store_true", help="Reserve one CPU core for general operations in order to not tank the instance.")
        parser.add_argument("--gpu_ids", nargs="+", type=int, help="Space-separated GPU IDs: e.g. 0, 0 1 2, etc. Not specifying this option will result in CPU-only operation.")

        self.initialized = True
        return parser

    def print_options(self, opt):
        """Print user-defined options to the command line, noting default values if different."""

        message = ""
        message += "----------------- Options ---------------\n"

        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)

            if v != default:
                comment = "\t[default: {}".format(str(default))

            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)

        message += "------------------- End -----------------"
        print(message)

    def option_file_path(self, opt, makedir=False):
        """Construct file path to read/write user options."""

        experiment_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if makedir:
            utils.mkdirs(experiment_dir)
        file_name = os.path.join(experiment_dir, "opt")

        return file_name

    def save_options(self, opt):
        """Write user options to file."""

        file_name = self.option_file_path(opt, makedir=True)

        with open(file_name + ".txt", "wt") as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ""
                default = self.parser.get_default(k)

                if v != default:
                    comment = "\t[default: %s]" % str(default)
                opt_file.write("{:>25}: {:<30}{}\n".format(str(k), str(v), comment))

        with open(file_name + ".pkl", "wb") as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        """Update existing options with ones from external file, if they exist."""

        new_opt = self.load_options(opt)

        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})

        return parser

    def load_options(self, opt):
        """Load a previously-saved set of options."""

        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + ".pkl", "rb"))

        return new_opt

    def gather_options(self):
        """Create an options parser, and pass it around various other parts of
        the codebase in order to dynamically add relevant options depending on
        which modules are loaded.
        """

        # Initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Get basic options
        opt, unknown = parser.parse_known_args()

        # Add/modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.is_train)

        # Add/modify dataset-related parser options
        dataset_type = opt.dataset_type
        dataset_option_setter = datasets.get_option_setter(dataset_type)
        parser = dataset_option_setter(parser, self.is_train)

        # Add/modify instance-related parser options
        parser = utils.add_instance_specs(parser, opt.reserve_one_core)
        opt, unknown = parser.parse_known_args()

        # If selected, load existing options from file
        # Previous options will be overwritten
        if opt.load_opt_from_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def parse(self, save=False):
        """Parse all options and print them out, ensuring batch_size won't break the experiment."""

        opt = self.gather_options()
        opt.is_train = self.is_train
        self.print_options(opt)

        if opt.is_train:
            self.save_options(opt)

        if opt.gpu_ids:
            torch.cuda.set_device(opt.gpu_ids[0])

        # Make sure batch_size doesn't break everything
        assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
            "Batch size {} won't work. It must be a multiple of # GPUs ({}).".format(
                opt.batch_size, len(opt.gpu_ids))

        self.opt = opt
        return self.opt