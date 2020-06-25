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
import json
from obb_anns import OBBAnns

@DATASETS.register_module
class DeepScoresV2Dataset(CocoDataset):


    def load_annotations(self, ann_file):
        self.obb = OBBAnns(ann_file)
        self.obb.load_annotations()
        self.obb.set_annotation_set_filter(['deepscores'])
        self.cat_ids = list(self.obb.get_cats().keys())
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.label2cat = {v: k for k, v in self.cat2label.items()}
        self.CLASSES = tuple([v["name"] for (k, v) in self.obb.get_cats().items()])
        self.img_ids = [id['id'] for id in self.obb.img_info]

        return self.obb.img_info

    def get_ann_info(self, idx):
        return self._parse_ann_info(*self.obb.get_img_ann_pair(idxs=[idx]))

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i, img_info in enumerate(self.obb.img_info):
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

    def prepare_json_dict(self, results):
        json_results = {"annotation_set": "deepscores", "proposals": []}
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['img_id'] = img_id
                    data['bbox'] = [str(nr) for nr in bboxes[i][0:-1]]
                    data['score'] = str(bboxes[i][-1])
                    data['cat_id'] = self.label2cat[label]
                    json_results["proposals"].append(data)
        return json_results

    def write_results_json(self, results, filename=None):
        if filename is None:
            filename = "deepscores_results.json"
        json_results = self.prepare_json_dict(results)

        with open(filename, "w") as fo:
            json.dump(json_results, fo)

        return filename

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 average_thrs=False):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        filename = self.write_results_json(results)

        self.obb.load_proposals(filename)
        metric_results = self.obb.calculate_metrics(iou_thrs=iou_thrs, classwise=classwise, average_thrs=average_thrs)
        print(metric_results)
        return metric_results