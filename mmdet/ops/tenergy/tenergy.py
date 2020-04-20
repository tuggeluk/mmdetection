import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from torch.autograd import Function
from torch.nn.modules.module import Module

from . import tenergy_cuda
#import tenergy_cuda

class TENERGYFunction(Function):

    @staticmethod
    def forward(ctx, masks, scale_factor, max_energy):
        assert scale_factor >= 1
        assert max_energy  > 1
    
        ctx.scale_factor = scale_factor
        ctx.mask_size = masks.size()

        n, c, h, w = masks.size()
        output = masks.new_zeros((n, c, h * scale_factor, w * scale_factor))
        if masks.is_cuda:
            tenergy_cuda.vote(masks,scale_factor, max_energy, output)
        else:
            raise NotImplementedError


        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        return None

tenergy_naive = TENERGYFunction.apply




