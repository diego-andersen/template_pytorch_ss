import os
import multiprocessing
import requests
import torch


def mkdirs(paths):
    """Make directories recursively if they don't exist."""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """Make a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def instance_is_terminating():
    """
    Ping the instance to check whether a termination event has been issued.
    If the resulting status code is anything other than 404, it means a stop,
    hibernate or terminate event has been issued.
    See: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html
    """
    status_code = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code
    return status_code != 404


def add_instance_specs(parser, reserve_one_core=False):
    """
    Detect hardware specs for machine this is running on and adjust opt accordingly.
    Optionally reserve one CPU core for general operation in case something else is running
    on the machine.
    """
    n_gpus = torch.cuda.device_count()
    n_cpus = multiprocessing.cpu_count()

    if n_gpus == 0:
        active_gpus = [-1]
    else:
        active_gpus = list(range(n_gpus))

    if reserve_one_core:
        n_cpus -= 1

    parser.set_defaults(gpu_ids=active_gpus)
    parser.set_defaults(n_threads=n_cpus)

    return parser


def save_network(net, label, epoch, opt):
    """Save model and weights to disk.

    Parameters:
        net : object (nn.Module subclass)
            Model class representing network architecture + weights.
        label : str
            Custom label to name file with.
        epoch : int
            Current training epoch, gets added to filename.
        opt : TrainOptions
            User options.
    """
    save_filename = '{}_net_{}.pth'.format(epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '{}_net_{}.pth'.format(epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net