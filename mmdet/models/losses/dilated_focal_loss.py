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
                 num_classes,
                 b=3,
                 factor=1.0,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """Dilated focal loss for extreme class imbalances.

        Args:
            num_classes (int): Number of classes to be regressed to.
            b (int): Length of one edge of the structuring element used for the
                dilation. If none is given, a structuring element with edge
                length 1 is used.
            factor (float or list): If factor is a single value,
                then this is the factor by which any value inside of the mask
                will be multiplied by. If factor is an array, it must have
                length C, where C is is the number of classes and is the factor
                by which the loss of the corresponding class inside their
                respective masks will be multiplied by.
            use_sigmoid (bool): Whether or not to use a sigmoidal function.
                Non-sigmoidal mode is currently not implemented
            gamma (float): Gamma value for focal loss.
            alpha (float): Alpha value for focal loss.
            reduction (str): One of 'mean', 'sum', or 'none'. Reduction used to
                do weight reduced loss.
            loss_weight (float): How much this loss matters to the whole loss.
        """
        super(DilatedFocalLoss, self).__init__(use_sigmoid, gamma, alpha,
                                               reduction, loss_weight)
        # Generate a proper structuring element from the given structure.
        # This is necessary since we only use a 2D structuring element since we
        # only want to dilate in 2D, i.e. not between classes. We therefore
        # require a 4D structuring element.
        structure = np.full((b, b), 1)
        zeros = np.zeros_like(structure)
        structure = np.stack((zeros, structure, zeros))
        zeros = np.zeros_like(structure)
        self.structure = np.stack((zeros, structure, zeros))

        self.num_classes = num_classes
        if isinstance(factor, float):
            self.factor = [factor for _ in range(num_classes)]
        else:
            self.factor = factor
            error_message = "List of factors must be the same length as " \
                            "the number of classes."
            assert len(factor) == num_classes, error_message

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Runs a forwards pass.

        Shapes:
            pred: [n, c, h, w]
            target: [n, h, w]
            weight: Irrelevant as long as weight.nelement() == target.nelement()
        Args:
            pred (torch.Tensor): Predictions from the network.
            target (torch.Tensor): Ground truth.
            weight (None or torch.Tensor): Element-wise weighting for each of
                the elements in the target.
            avg_factor (float): Average factor when computing the weight
                reduced loss.
            reduction_override (None or str): Overrides the reduction
                specified when initializing the loss instance.
        """
        # Assertions
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert len(pred.shape) == 4, 'pred must have the shape [n, c, h, w]'

        # Ensure that factor is a torch.Tensor object on the correct device
        if isinstance(self.factor, (list, tuple)):
            self.factor = torch.tensor(self.factor, dtype=torch.float,
                                       device=pred.device)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            # First generate the mask
            mask = self.dilated_mask(target)

            mask = (mask.permute(0, 2, 3, 1)
                    * self.factor).permute(0, 3, 1, 2)

            # Change all 0s to 1s
            # Using torch.where() is about twice as fast as using x[x == 0]
            ones = torch.full_like(mask, 1)
            mask = torch.where(mask == 0, ones, mask)

            mask = mask.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

            # Flatten since _sigmoid_focal_loss requires flat
            flat_pred = pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            flat_targets = target.reshape(-1).long()

            loss_cls = _sigmoid_focal_loss(flat_pred,
                                           flat_targets,
                                           self.gamma,
                                           self.alpha)
            loss_cls *= mask

            # Do weight reduction
            if weight is not None:
                weight = weight.view(-1, 1)
            loss_cls = weight_reduce_loss(loss_cls, weight, reduction,
                                          avg_factor)
            return loss_cls

    def dilated_mask(self, target):
        """Generates a 2D numpy binary mask that is then dilated.

        Args:
            target (torch.Tensor): Target in the shape of [n, h, w]

        Returns:
            torch.Tensor: The dilated mask in the shape [n, c, h, w]
        """
        # Generate one-hot encoding in the shape (n, c, h, w)
        target_onehot = torch.zeros((self.num_classes + 1, *target.shape))
        target_onehot.scatter_(0, target.unsqueeze(0).long().detach().cpu(), 1)
        # Get rid of the onehot of the background class
        target_onehot = target_onehot[1:].permute(1, 0, 2, 3)

        # Now dilate it
        dilated = binary_dilation(target_onehot, self.structure).astype(float)
        return torch.tensor(dilated, dtype=target.dtype, device=target.device)
