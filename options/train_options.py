from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Add training-specific options to global options. Should be instantiated directly in train.py."""
    def initialize(self, parser):
        super().initialize(parser)

        # I/O settings
        parser.add_argument("--print_freq", type=int, default=1, help="Frequency (in steps) for printing training results to std out.")
        parser.add_argument("--save_latest_freq", type=int, default=10, help="Number of training steps between every 'latest' checkpoint. These overwrite each other.")
        parser.add_argument("--save_epoch_freq", type=int, default=1, help="Number of epochs between every checkpoint. These do not get overwritten.")

        parser.add_argument("--continue_training", action="store_true", help="Continue training: load the latest model, epoch and step.")
        parser.add_argument("--which_epoch", type=str, default="latest", help="Which epoch to load? Set to 'latest' to use latest cached model.")

        # NOTE: total epochs = n_epochs + n_epochs_decay
        parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs to train for with fixed learning rate.")
        parser.add_argument("--n_epochs_decay", type=int, default=50, help="Number of epochs to train for with decaying learning rate.")

        parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Which optimizer to use.")
        parser.add_argument("--loss_function", type=str, default="cross_entropy", choices=["cross_entropy", "dice", "focal"],
            help="Which criterion/loss function to use.")

        # Hyperparameters go here
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Optimizer learning rate.")
        parser.add_argument("--beta1", type=float, default=0.9, help="Coefficient used for computing running averages of gradient and its square.")
        parser.add_argument("--beta2", type=float, default=0.999, help="Coefficient used for computing running averages of gradient and its square.")

        self.is_train = True
        return parser
