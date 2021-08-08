import torch.optim as optim

import models
from models.sync_batchnorm import DataParallelWithCallback
from utils.optimizers import get_optimizer

class StandardTrainer():
    """
    Trainer creates the model and optimizers, and uses them to updates the weights of the network
    while reporting losses and the latest visuals to visualize the progress in training.

    NOTE: For multi-GPU operation, https://github.com/vacancy/Synchronized-BatchNorm-PyTorch is required as a dependency.
    Clone it into models/networks and extract the sync_batchnorm directory into networks.

    Attributes
    ----------
    opt : object (TrainingOptions)
        User options.
    model : object
        Model class from models directory, represents the network.
    model_on_one_gpu :
    optimizer :
    outputs : object (torch.Tensor)
        Single batch of outputs from the model.
    losses : object (torch.Tensor)
        Single batch of losses from the model.
    old_lr : float
        Current learning rate.
    lr_decay : float
        Learning rate decay fraction.
    """

    def __init__(self, opt):
        self.opt = opt
        self.model = models.create_model(opt)
        self.criterion = models.create_loss_function(opt)
        self.old_lr = opt.learning_rate
        self.lr_decay = opt.learning_rate / opt.n_epochs_decay

        # Prep for multi-GPU
        if len(opt.gpu_ids) > 1:
            self.model = DataParallelWithCallback(self.model, device_ids=opt.gpu_ids)
            self.model_on_one_gpu = self.model.module
        else:
            self.model_on_one_gpu = self.model

        self.optimizer = get_optimizer(opt, self.model_on_one_gpu)
        self.outputs = None

    def run_model_one_step(self, data):
        """One full iteration through training loop.

        1. Zero out gradients
        2. Push data through network graph
        3. Calculate losses
        4. Back-propagate, step forward through loss function.
        """
        self.optimizer.zero_grad()
        inputs = data['image']
        targets = data['label']
        outputs = self.model(inputs)
        losses = self.criterion(outputs, targets)
        loss = sum(losses.values()).mean()
        loss.backward()
        self.optimizer.step()
        self.losses = losses
        self.outputs = outputs

    def get_latest_losses(self):
        return self.losses

    def get_latest_outputs(self):
        return self.outputs

    def update_learning_rate(self, epoch):
        if epoch > self.opt.n_epochs:
            new_lr = self.old_lr - self.lr_decay

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            print("Updating learning rate: {:f} -> {:f}".format(self.old_lr, new_lr))
            self.old_lr = new_lr

    def save(self, epoch):
        self.model_on_one_gpu.save(epoch)

    def terminate_training(self, epoch):
        print("Terminating training after epoch {}".format(epoch))
        exit(0)

