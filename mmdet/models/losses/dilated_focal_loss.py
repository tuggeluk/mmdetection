"""Dilated Focal Loss.

Class-wise loss used for cases with extreme class imbalances.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    February 11, 2020
"""
from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss
from ..losses import FocalLoss
import numpy as np
import torch
from scipy.ndimage import binary_dilation


@LOSSES.register_module
class DilatedFocalLoss(FocalLoss):
    def __init__(self,
                 b=3,
                 factor=1.0,
                 num_classes=None,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """Dilated focal loss for extreme class imbalances.

        Args:
            b (int): Length of one edge of the structuring element used for the
                dilation. If none is given, a structuring element with edge
                length 1 is used.
            factor (float or torch.Tensor): If factor is a single value,
                then this is the factor by which any value inside of the mask
                will be multiplied by. If factor is an array, it must have
                length C, where C is is the number of classes and is the factor
                by which the loss of the corresponding class inside their
                respective masks will be multiplied by.
            num_classes (int): Number of classes to be regressed. Only
                required if multiclass factors are wanted, i.e. factor is not a
                float.

        """
        super(DilatedFocalLoss, self).__init__(use_sigmoid, gamma, alpha,
                                               reduction, loss_weight)
        # Generate a proper structuring element from the given structure.
        # This is necessary since we need to create a 4D structuring element,
        # but a 2D binary structure is used. A 2D structure is used since we
        # only want to dilate in 2D, i.e. not between classes.
        structure = np.full((b, b), 1)
        zeros = np.zeros_like(structure)
        structure = np.stack((zeros, structure, zeros))
        zeros = np.zeros_like(structure)
        self.structure = np.stack((zeros, structure, zeros))

        self.factor = factor
        if isinstance(factor, float):
            self.multiclass = False
        else:
            assert isinstance(num_classes, int), 'num_classes must be given ' \
                                                 'if multiclass is used.'
            self.multiclass = True
            self.num_classes = num_classes

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert len(pred.shape) == 4, 'pred must have the shape [n, c, h, w]'
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            # First generate the mask
            if self.multiclass:
                mask = self.multiclass_dilated_mask(target.detach().cpu())
            else:
                mask = self.dilated_mask(target.detach().cpu())

            mask = mask.to(device=target.device)

            if self.multiclass:
                mask = (mask.permute(0, 2, 3, 1)
                        * self.factor).permute(0, 3, 1, 2)
            else:
                mask *= self.factor

            # Change all 0s to 1s
            # Using torch.where() is about twice as fast as using x[x == 0]
            ones = torch.full_like(mask, 1)
            mask = torch.where(mask == 0, mask, ones)
            loss_cls = _sigmoid_focal_loss(pred, target, self.gamma, self.alpha)
            loss_cls *= mask

            # Do weight reduction
            if weight is not None:
                weight = weight.view(-1, 1)
            loss_cls = weight_reduce_loss(loss_cls, weight, reduction,
                                          avg_factor)
            return loss_cls

    def dilated_mask(self, target):
        """Generates a 2D numpy binary mask and then dilates it"""
        # First generate a [n, 1, h, w] tensor
        np_target = target.numpy()
        # Then return the dilation of that tensor by the structuring element
        dilated = binary_dilation(np_target, self.structure).astype(float)
        return torch.tensor(dilated)

    def multiclass_dilated_mask(self, target):
        """Generates a 2D numpy binary mask that can then be dilated.

        Returns:
            torch.Tensor
        """
        # Generate one-hot encoding in the shape (n, c, h, w)
        target_onehot = torch.zeros((self.num_classes, target.shape[0],
                                     target.shape[1], target.shape[2]))
        target_onehot.scatter_(0, target.unsqueeze(0), 1)
        target_onehot = target_onehot.permute(1, 0, 2, 3)

        # Now dilate it
        dilated = binary_dilation(target_onehot, self.structure).astype(float)
        return torch.tensor(dilated)


