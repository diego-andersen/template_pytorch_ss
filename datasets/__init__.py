import importlib
import torch.utils.data

from .base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """
    Import "datasets/dataset_name_dataset.py" and return the dataset's class.
    Enables the use of --dataset_type [name] option in BaseOptions.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'

    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if not dataset:
        raise ValueError(
            "Dataset {} not found inside {} folder".format(target_dataset_name, dataset_filename))

    return dataset


def get_option_setter(dataset_name):
    """
    Return a dataset's option setter, in order to set/modify dataset-specific
    options and add them to global options.
    """
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt):
    """Create a Dataloader object based on a dataset specified in options."""
    dataset = find_dataset_using_name(opt.dataset_type)
    instance = dataset()
    instance.initialize(opt)
    print("Dataset [{}] of size {} was created".format(
          type(instance).__name__, len(instance)))

    # Note that the dataloader drops the final batch when training
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.n_threads,
        drop_last=opt.is_train
    )

    return dataloader
