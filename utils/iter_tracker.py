import os
import numpy as np


class IterationTracker():
    """
    Helper class that keeps track of training iterations.

    Can augment this with time tracking if needed.
    NOTE: When saving epochs, the class writes current iteraion + 1 to file,
    so that training can resume from next epoch.

    Attributes:
        opt : BaseOptions subclass
            User options.
        dataset_size : int
            Number of samples in the dataset.
        first_epoch : int
            Number of first training epoch.
        current_epoch : int
            Number of current training epoch.
        total_epochs : int
            Total number of training epochs, includes ones with decaying learning rate.
        epoch_step : int
            Current training steps taken within each epoch, resets to zero when epoch changes.
        iter_record_path : path-like object
            Text file in which to record current training epoch/step.
        global_step :int
            Total number of training steps taken, across all epochs.
    """
    def __init__(self, opt, dataset_size):
        self.opt = opt
        self.dataset_size = dataset_size

        self.first_epoch = 1
        self.total_epochs = opt.n_epochs + opt.n_epochs_decay
        self.global_step = 1
        self.epoch_step = 1
        self.iter_record_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'iter.txt')

        # Overwrite starting epoch/step if resuming
        if opt.continue_training:
            try:
                self.first_epoch, self.epoch_step, self.global_step = np.loadtxt(
                    self.iter_record_path, delimiter=',', dtype=int)
                self.record_one_iteration()
                print('Resuming from epoch {:d} at step {:d} ({:d})'.format(self.first_epoch, self.epoch_step, self.global_step))
            except:
                print('Could not load iteration record at {}. Starting from beginning.'.format(
                      self.iter_record_path))

    def training_epochs(self):
        """Return an iterator over the training epochs."""
        return range(self.first_epoch, self.first_epoch + self.total_epochs + 1)

    def record_epoch_start(self, epoch):
        self.epoch_step = 1
        self.current_epoch = epoch

    def record_epoch_end(self):
        if self.current_epoch % self.opt.save_epoch_freq == 0:
            np.savetxt(self.iter_record_path, (self.current_epoch, 1, self.global_step),
                       delimiter=',', fmt='%d')
            print('Saved current iteration count at {}.'.format(self.iter_record_path))

    def record_one_iteration(self):
        self.global_step += 1
        self.epoch_step += 1

    def record_current_iter(self):
        np.savetxt(self.iter_record_path, (self.current_epoch, self.epoch_step, self.global_step),
                   delimiter=',', fmt='%d')
        print('Saved current iteration count at {}.'.format(self.iter_record_path))

    def total_samples(self):
        return self.global_step * self.opt.batch_size

    def needs_saving(self):
        return (self.global_step % self.opt.save_latest_freq) == 0

    def needs_printing(self):
        return (self.global_step % self.opt.print_freq) == 0
