"""Watershed-FCOS Head

A head that uses principles from both the Deep Watershed Detector paper as well
as the FCOS paper.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    Lukas Tuggener <tugg@zhaw.ch>

Created on:
    January 22, 2020
"""
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from .wfcos_head import WFCOSHead
from mmdet.ops import ConvModule, Scale
# Visualization imports
import debugging.visualization_tools as vt
from mmcv.visualization import imshow_det_bboxes
from mmdet.core import tensor2imgs
import numpy as np

INF = 1e8


@HEADS.register_module
class WFCOSHead_regression(WFCOSHead):
    def __init__(self, *args, **kwargs):
        """
        See WFCOS Head for Documentation, implements some experimental features
        - energy regression in the style of FCOS
        - r value based on bbox size
        - r values per feature stack
        - "smooth" energy
        """
        print("call super")
        super(WFCOSHead_regression, self).__init__(*args, **kwargs)



