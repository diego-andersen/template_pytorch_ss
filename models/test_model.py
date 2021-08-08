import torch.nn as nn

from models.base_model import BaseModel


class TestModel(BaseModel):
    """Basic semantic segmentation model containing one of each operation.

    Use it to check if the rest of the boilerplate works.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add model-specific CLI options/defaults here."""
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.net_in = self._conv_in(3, 64, kernel_size=3, stride=1, padding=1)
        self.down = self._conv_down(64, 128, kernel_size=3, stride=1, padding=1)
        self.up = self._conv_up(128, 64, kernel_size=2, stride=2)
        self.net_out = self._conv_out(64, opt.n_classes)

    def _conv_in(self, ch_in, ch_out, **kwargs):
        return nn.Conv2d(ch_in, ch_out, **kwargs)

    def _conv_down(self, ch_in, ch_out, **kwargs):
        return nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ch_in, ch_out, **kwargs),
            nn.ReLU(inplace=True)
        )

    def _conv_up(self, ch_in, ch_out, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, **kwargs),
            nn.ReLU(inplace=True)
        )

    def _conv_out(self, ch_in, ch_out):
        return nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        x = self.net_in(x)
        x = self.down(x)
        x = self.up(x)
        x = self.net_out(x)

        return x
