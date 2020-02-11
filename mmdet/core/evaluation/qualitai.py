"""QualitAI.

Implements validation according to binary classification metrics for QualitAI.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    February 06, 2020
"""
import torch

import mmcv
from mmcv.parallel import collate, scatter
from mmdet.core.evaluation import DistEvalHook
from mmcv.runner import Hook
from torch.utils.data import Dataset
from os.path import split, join

from mmdet import datasets


class QualitaiDistEvalHook(DistEvalHook):
    def evaluate(self, runner, results):
        device = results.device
        gt = torch.zeros(len(self.dataset)).to(dtype=torch.bool, device=device)
        dataset = self.dataset.coco

        for i, result in enumerate(results):
            data = runner.l
            path, img_name = split(runner.loader)

# Do inference
#         results = np.zeros((len(self.loader), 99), dtype=bool)
#         gt = np.zeros(len(self.loader), dtype=bool)
#         prog_bar = ProgressBar(len(self.loader.dataset))
#
#         file_id_lookup = self.get_ids_of_files(self.dataset.coco)
#
#         for i, data in enumerate(self.loader):
#             with torch.no_grad():
#                 result = model(return_loss=False, rescale=True, **data)
#
#             # This entire block is just to get the img_id
#             path, img_name = split(data['img_meta'][0].data[0][0]['filename'])
#             if img_name in file_id_lookup:
#                 img_id = file_id_lookup[img_name]
#             else:
#                 img_name = join(split(path)[1], img_name)
#                 if img_name in file_id_lookup:
#                     img_id = file_id_lookup[img_name]
#                 else:
#                     raise KeyError(img_name)
#
#             # This is to get the ground truth binary classifications
#             gt[i] = len(self.dataset.coco.getAnnIds(img_id)) > 0
#
#             # This is to get the predicted binary classifications
#             for thr in range(99):
#                 results[i, thr] = np.count_nonzero(
#                     result[0][:,4] > thr / 100
#                 ) > 0
#
#             img_shape = data['img_meta'][0].data[0][0]['ori_shape']
#             bool_pred = self.transform_preds_to_boolean(
#                 img_shape[0:2],
#                 result[0]
#             )
#
#             # Write out images #################################################
#             if i % 30 == 0:
#                 bool_target = self.transform_targets_to_boolean(
#                     self.dataset.coco, img_id, img_shape[0:2])
#
#                 target_img = np.zeros(img_shape, dtype='uint8')
#                 target_img[bool_target] = [0, 255, 0]
#                 target_img = Image.fromarray(target_img)
#                 pred_img = np.zeros(img_shape, dtype='uint8')
#                 pred_img[bool_pred] = [255, 0, 0]
#                 pred_img = Image.fromarray(pred_img)
#
#                 ori_img = Image.open(data['img_meta'][0].data[0][0]['filename'])
#                 ori_img.save('/workspace/outputs/{}-ori.jpg'.format(i))
#                 target_img.save('/workspace/outputs/{}-target.png'.format(i))
#                 pred_img.save('/workspace/outputs/{}-pred.png'.format(i))
#             # END ##############################################################
#             prog_bar.update()
#
#         # Calculate values
#         "\nWriting out results..."
#
#         print('\nStarting evaluation according to QualitAI metrics...')
#
#         print('in copyable csv:\n')
#         print('thr,acc,precision,recall')
#         per_thr_acc = []
#         for i in range(99):
#             acc = metrics.balanced_accuracy_score(gt, results[:, i])
#             per_thr_acc.append(acc)
#             precision = metrics.precision_score(gt, results[:, i])
#             recall = metrics.recall_score(gt, results[:, i])
#             print("{:.2f},{:.5f},{:.5f},{:.5f}"
#                   .format((i + 1) / 100, acc, precision, recall))