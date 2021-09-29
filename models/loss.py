import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add model-specific CLI options/defaults here."""
        parser.add_argument("--loss_ignore_index", type=int, default=None, help="Index value that is ignored when calculating input gradients.")
        parser.add_argument("--loss_reduction", type=str, help="Specifies reduction to apply to loss output.",
            choices=["none", "mean", "sum"])

        return parser

    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super().__init__()
        self.loss_function =  nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index, reduction=reduction)

    def forward(self, outputs, targets):
        loss = self.loss_function(outputs, targets)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, outputs, targets):
        if self.ignore_index not in range(targets.min(), targets.max()):
            if (targets == self.ignore_index).sum() > 0:
                targets[targets == self.ignore_index] = targets.min()
        targets = make_one_hot(targets.unsqueeze(dim=1), classes=outputs.size()[1])
        outputs = F.softmax(outputs, dim=1)
        output_flat = outputs.contiguous().view(-1)
        target_flat = targets.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2.0 * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, outputs, targets):
        logpt = self.CE_loss(outputs, targets)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()
