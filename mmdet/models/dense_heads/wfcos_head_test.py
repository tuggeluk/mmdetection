import unittest

import torch

from mmdet.models import WFCOSHead

INF = 1e8

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.wfcos_head = WFCOSHead(
            num_classes=136,
            in_channels=256,
            max_energy=20,
            feat_channels=256,
            stacked_convs=4,
            strides=(8, 16, 32, 64, 128),
            regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
            loss_cls={'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 1.0},
            loss_bbox={'type': 'IoULoss', 'loss_weight': 1.0},
            loss_energy={'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'loss_weight': 1.0},
            conv_cfg=None,
            norm_cfg=None,
            split_convs=False,
            assign="min_edge",
            r=5.0,
            mask_outside=False,
            bbox_portion=1,
            kwargs={
                'train_cfg': {
                    'assigner': {
                        'type': 'MaxIoUAssigner',
                        'pos_iou_thr': 0.5,
                        'neg_iou_thr': 0.4,
                        'min_pos_iou': 0,
                        'ignore_iof_thr': -1
                    },
                    'allowed_border': -1,
                    'pos_weight': -1,
                    'debug': False
                },
                'test_cfg': {
                    'nms_pre': 1000,
                    'min_bbox_size': 0,
                    'score_thr': 0.3,
                    'nms': {
                        'type': 'nms',
                        'iou_thr': 0.2
                    },
                    'max_per_img': 1000
                }
            }
        )

    def test_energy_is_same(self):
        feat_dim = torch.Size([100, 100])
        img_size = (800, 800, 3)
        a_bboxes = torch.tensor([[452.7114, 323.7556, 454.4020, 408.2842, 41.0000],
                               [452.7114, 573.9602, 454.4020, 692.3002,  41.0000],
                               [401.9912, 185.1288, 403.6819, 259.5139,  41.0000],
                               [478.0714, 185.1288, 479.7621, 259.5139,  41.0000],
                               [117.9587, 577.3413, 119.6494, 692.3002,  41.0000],
                               [298.8604, 185.1288, 300.5510, 254.4423,  41.0000],
                               [117.9587, 372.7822, 119.6494, 435.3334,  41.0000],
                               [141.6281, 195.2722, 143.3187, 267.9668,  41.0000],
                               [655.5918, 185.1288, 657.2825, 254.4423,  41.0000],
                               [658.9730, 274.7291, 660.6637, 360.9482,  41.0000]], device='cuda:0')
        o_bboxes = torch.tensor([[553.0000, 528.5000, 313.0000, 527.2500, 312.9183, 542.9268, 552.9183, 544.1768, 122.0000],
                                [239.7677, 707.2427, 183.3551, 703.3521, 182.8018, 711.3743, 239.2145, 715.2649, 121.0000],
                                [314.4858, 282.6746, 295.9953, 277.3914, 285.8066, 313.0518, 304.2971, 318.3350,  87.0000],
                                [574.2500, 514.7500, 565.5000, 497.2500, 539.0000, 510.5000, 547.7500, 528.0000,  28.0000],
                                [125.0001, 515.0000, 116.5001, 498.0000,  92.0000, 510.2500, 100.5000, 527.2500,  24.0000],
                                [126.0000, 668.2500, 117.5001, 651.2500,  93.5000, 663.2500, 102.0000, 680.2500,  24.0000],
                                [267.7501, 658.0000, 259.2500, 641.0000, 234.2500, 653.5000, 242.7501, 670.5000,  26.0000],
                                [374.2500, 319.7500, 365.7500, 302.7500, 340.7500, 315.2500, 349.2500, 332.2500,  24.0000],
                                [268.0000, 524.7500, 259.5000, 507.7500, 234.0001, 520.5000, 242.5001, 537.5000,  26.0000],
                                [629.7500, 278.2500, 621.2500, 261.2500, 596.7500, 273.5000, 605.2500, 290.5000,  24.0000]],
                                device='cuda:0')
        bboxes = a_bboxes
        energy_old = self.wfcos_head._get_energy_single_old(feat_dim, img_size, bboxes)
        energy_new = self.wfcos_head._get_energy_single_new(feat_dim, img_size, bboxes)
        self.assertEqual(energy_new, energy_old)


if __name__ == '__main__':
    unittest.main()
