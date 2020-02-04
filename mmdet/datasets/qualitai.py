"""DEEPSCORES

Provides access to the QualitAI database with a COCO-like interface. The
only changes made compared to the coco.py file are the class labels.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 23, 2019
"""
from .coco import *


@DATASETS.register_module
class QualitaiDataset(CocoDataset):

    CLASSES = ('bad')
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            # if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
            #     continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds
