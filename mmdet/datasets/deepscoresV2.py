"""DEEPSCORESV2

Provides access to the DEEPSCORESV2 database with a COCO-like interface. The
only changes made compared to the coco.py file are the class labels.

Author:
    Lukas Tuggener <tugg@zhaw.ch>
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 23, 2019
"""
from .coco import *
import os
from obb_anns import OBBAnns

@DATASETS.register_module
class DeepScoresV2Dataset(CocoDataset):


    def load_annotations(self, ann_file):
        self.obb = OBBAnns(ann_file)
        self.obb.load_annotations()
        self.cat_ids = self.obb.get_cats().keys()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        DeepScoresV2Dataset.CLASSES = tuple([v["name"] for (k, v) in self.obb.get_cats().items()])

        return self.obb.img_info

    def get_ann_info(self, idx):
        return self._parse_ann_info(*self.obb.get_img_ann_pair(idxs=[idx]))

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and len(img_info['ann_ids']) == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        img_info, ann_info = img_info[0], ann_info[0]
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        for i, ann in ann_info.iterrows():
            # we have no ignore feature
            if ann['area'] <= 0:
                continue

            bbox = ann['a_bbox']
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['cat_id'][0]])

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=None,
            seg_map=None)
        return ann
