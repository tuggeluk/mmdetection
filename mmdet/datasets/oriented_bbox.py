"""Oriented BBox Dataset.

Provides a base class for datasets with oriented bounding boxes.

Description

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    February 19, 2020
"""
import json
from time import time
import numpy as np
import os.path as osp
from PIL import Image
from matplotlib.path import Path

from .custom import CustomDataset


class OrientedBBoxesDataset(CustomDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        """Base class for datasets with oriented bounding boxes.

        Annotation schema:
            {
                "info": {
                    "description": (str) description,
                    "version": (str) version number,
                    "year": (int) year released,
                    "contributor": (str) contributor,
                    "date_created": (str) "YYYY/MM/DD"
                },
                "categories": {
                    "cat_id": (str) category_name,
                    ...
                },
                "images": [
                    {
                        "id": (int) n,
                        "filename": (str) 'file_name.jpg',
                        "width": (int) x,
                        "height": (int) y,
                        "ann_ids": [(int) ann_ids]
                    },
                    ...
                ],
                "annotations": {
                    "ann_id": {
                        "bbox": (list of floats) [x1, y1,..., x4, y4],
                        "cat_id": (int) cat_id,
                        "area": (float) area in pixels,
                        "img_id": (int) img_id
                    },
                    ...
                }
            }

        Notes:
            - The annotation file is in JSON format.
            - The 'annotations' field is optional for testing.
            - Segmentations are found in a png file named '[filename]_seg.png'
                - The segmentation file is a grayscale 8-bit png image where the
                    pixel values correspond to the cat_id.
                - If more categories are required, alternative mappings can be
                    defined by overriding the _parse_ann_info method.
            - cat_id, and ann_id are stringified ints.
            - cat_id, ann_id, and img_id starts at 1.

        Proposal schema:
            {
                "proposals": [
                    {
                        "bbox": (list of floats) [x1, y1,..., x4, y4],
                        "cat_id": (int) cat_id,
                        "img_id": (int) img_id
                    },
                    ...
                ]
            }

        Notes:
            - The proposals file is in JSON format.
            - bbox is in the same format as for annotations
            - A check is done to make sure all img_idxs and cat_ids that are
                referred to in the proposal file is in the annotation file to
                make sure that the proposals corresponds to the correct
                annotations file.
        """
        print('loading annotations...')
        # Setting up timer
        start_time = time()
        self.dataset_info = None
        self.cat_infos = None
        self.annotations = None

        super(OrientedBBoxesDataset, self).__init__(
            ann_file, pipeline, data_root, img_prefix, seg_prefix,
            proposal_file, test_mode, filter_empty_gt)

        # We create an image lookup table so we can quickly get the image
        # index by image id
        self.image_idx_lookup = {info['id']: i
                                 for i, info in enumerate(self.img_infos)}

        print("done! t={:.2f}s".format(time() - start_time))

    def __repr__(self):
        information = "<Oriented Bounding Box Dataset.\n"
        information += f"Data root: {self.data_root}\n"
        information += f"Ann file: {self.ann_file}\n"
        information += f"Num images: {len(self.image_infos)}\n"
        information += f"Num anns: {len(self.annotations)}\n"
        information += f"Num cats: {len(self.cat_infos)}"
        if self.proposal_file:
            information += f"\nProposal file: {self.proposal_file}"
            information += f"\nNum proposals: {len(self.proposals)}>"

        return information

    def load_annotations(self, ann_file):
        """Loads annotations into memory.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: A list of images.
        """
        with open(ann_file, 'r') as ann_file:
            data = json.load(ann_file)

        self.dataset_info = data['info']

        self.cat_infos = dict()
        for k, v in data['categories'].items():
            self.cat_infos[int(k)] = v
            self.CLASSES.append(v)

        self.annotations = {int(k): v for k, v in data['annotations'].items()}

        image_info = data['images']

        return image_info

    def load_proposals(self, proposal_file):
        """Gets proposals from file.

        Args:
            proposal_file (str): Path to the proposal file.

        Returns:
            list: List of proposals.
        """
        with open(proposal_file, 'r') as p_file:
            props = json.load(p_file)['proposals']

        proposals = {i: [] for i in range(len(self.img_infos))}
        for prop in props:
            prop_img_idx = self.image_idx_lookup[prop["img_id"]]
            proposals[prop_img_idx].append(prop)

        return proposals

    def get_ann_ids(self, img_ids=None, cat_ids=None):
        """Gets annotations of a certain image or category.

        If both img_ids and cat_ids are given, then it's filtered first by
        img_ids then by cat_ids. If none are given, returns all the keys

        Args:
            img_idxs (list or tuple): List of img_idxs to find.
            cat_ids (list or tuple): List of cat_ids to find.

        Returns:
            list: List of ann_ids given the filters.
        """
        if len(img_ids) == len(cat_ids) == 0 \
                or (img_ids is None and cat_ids is None):
            return list(self.annotations.keys())

        filtered_anns = []
        if img_ids is not None:
            # Filter first by img_idxs
            for img_id in img_ids:
                img_idx = self.image_idx_lookup[int(img_id)]
                filtered_anns.extend(
                    self.image_info[img_idx]['ann_ids']
                )

        if cat_ids is not None:
            if img_ids is not None:
                # Then get the cats for those anns already filtered
                cat_filtered_anns = []
                for ann in filtered_anns:
                    if self.annotations['cat_id'] in cat_ids:
                        cat_filtered_anns.append(ann)
                filtered_anns = cat_filtered_anns

            else:
                # Go through all anns
                for k, v in self.annotations.items():
                    if v['cat_ids'] in cat_ids:
                        filtered_anns.append(k)
        return filtered_anns

    def get_ann_info(self, idx):
        """Gets annotation info for a given image index.

        Args:
            idx (int): The index of the image being requested.

        Returns:
            dict: Annotations as a dict with the following keys:
                bboxes (np.ndarray): (n, 8) bbox array.
                bboxes_ignore (np.ndarray): (0, 8) array. Always returned
                    empty since not sure what to put in here.
                labels (list[str]): Label names corresponding to each bbox.
                masks (np.ndarray): (n, h, w) binary masks with the extracted
                    segmentation masks where n is the index of the bbox.
                seg_map (str): Path to the segmentation mask.
        """
        gt_bboxes = []
        gt_labels = []
        gt_masks = []

        img_info = self.img_infos[idx]
        seg_map = osp.splitext(img_info['filename'])[0] + '_seg.png'
        seg_map = osp.join(self.seg_prefix, seg_map)
        full_seg_mask = np.array(Image.open(seg_map))  # Stored in (h, w)
        h, w = full_seg_mask.shape

        # Make an image level points grid
        # We create a meshgrid, which is flattened, combined into a
        # single array to get an array of points, then transposed to get
        # an (n, 2) array of points
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        points = np.array((X.flatten(), Y.flatten())).transpose()

        for ann in self._get_img_anns(idx):
            bbox = np.array(ann['bbox'])
            gt_bboxes.append(np.array(bbox))
            gt_labels.append(self.cat_infos[ann['cat_id']])

            bbox = bbox.reshape(4, 2)

            # Now to get the seg mask.
            # First get the axis aligned bounding box points
            x = (min(bbox[:, 0]), max(bbox[:, 0]))
            y = (min(bbox[:, 1]), max(bbox[:, 1]))

            # Now get all grid points within the axis aligned bounding box
            bbox_points_idx = []
            for y_pos in range(y[0], y[1]):
                bbox_points_idx.append(
                    np.r_[x[0] + (y_pos * w): x[1] + (y_pos * w)]
                )
            bbox_points_idx = np.concatenate(bbox_points_idx)

            # Then get the polygon as a path and check if the points are
            # within the polygon
            bbox_path = Path(bbox)
            bbox_mask = bbox_path.contains_points(points[bbox_points_idx])

            # Create a mask of the entire image whose value is true where the
            # bbox_mask is true after offsetting it to the proper position
            mask = np.zeros(points.shape[0], dtype=bool)
            mask[bbox_points_idx] = bbox_mask

            # Reshape the mask to be the same shape as the input image.
            mask = mask.reshape((h, w))

            # Now use the mask to selectively look at pixels in the input seg
            # image and find those whose values are the same as the category
            # of the annotation
            seg_mask = np.zeros(full_seg_mask.shape, dtype=bool)
            seg_mask[mask] = np.where(full_seg_mask[mask] == ann['cat_id'],
                                      True, False)
            gt_masks.append(seg_mask)

        return {'bboxes': np.stack(gt_bboxes),
                'bboxes_ignore': np.empty((0, 8)),
                'labels': gt_labels,
                'masks': np.stack(gt_masks),
                'seg_map': seg_map}


    def _get_img_anns(self, img_idx):
        ann_ids = self.img_infos[img_idx]['ann_ids']
        return [self.annotations[ann_id] for ann_id in ann_ids]
