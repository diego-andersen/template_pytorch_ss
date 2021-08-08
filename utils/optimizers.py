import torch.optim as optim

def get_optimizer(opt, model):
    param_group = list(model.parameters())

    if opt.optimizer == "adam":
        optimizer = optim.Adam(
            param_group,
            lr=opt.learning_rate,
            betas=(opt.beta1, opt.beta2)
            )
    elif opt.optimizer == "sgd":
        optimizer = optim.SGD(
            param_group,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay
            )
    else:
        print("Unknown optimizer: {}".format(opt.optimizer))

    return optimizer