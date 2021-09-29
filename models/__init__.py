import importlib
import torch


def find_model_using_name(model_name):
    """
    Import "models/modelname_model.py" and return the model's class.
    Enables the use of --model [name] option in BaseOptions.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # Note that model name is case-insensitive
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        print("Could not find model {} in {}.py.".format(target_model_name, model_filename))
        exit(0)

    return model


def get_option_setter(model_name):
    """
    Return a model's option setter, in order to set/modify model-specific
    options and add them to global options.
    """
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Instantiate a model object based on a model type specified in options."""
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("Model [{}] was created".format(type(instance).__name__))

    return instance


def find_loss_fn_using_name(loss_name):
    """
    Import "models/loss.py" and return the loss function's class.
    Enables the use of --loss [name] option in TrainOptions.

    TODO: Refactor this so that find_model_using_name() is the same function.
    """
    losslib = importlib.import_module("models.loss")

    # Note that model name is case-insensitive
    loss = None
    target_name = loss_name.replace('_', '') + 'loss'

    for name, cls in losslib.__dict__.items():
        if name.lower() == target_name.lower() \
           and issubclass(cls, torch.nn.Module):
            loss = cls

    if loss is None:
        print("Could not find loss function {} in {}.py.".format(target_name, loss_name))
        exit(0)

    return loss


def create_loss_function(opt):
    loss_fn = find_loss_fn_using_name(opt.loss_function)
    instance = loss_fn(opt)
    print("Loss function [{}] created".format(type(instance).__name__))

    return instance
