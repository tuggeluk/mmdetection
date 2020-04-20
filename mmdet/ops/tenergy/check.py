import os.path as osp
import sys

import mmcv
import torch

sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from tenergy import tenergy_naive  



mask = torch.randn(
    2, 100, 6, 6, requires_grad=False, device='cuda:0').sigmoid().double()


masks = torch.randn(
    2, 25, 200, 200, requires_grad=False, device='cuda:0').sigmoid().float()

time_n= 0

timer = mmcv.Timer()

x = tenergy_naive(masks.clone(),1, 60)
torch.cuda.synchronize()
time_n += timer.since_last_check()

print('\TENERGY time: {} ms/iter'.format(
    (time_n + 1e-3) ))


