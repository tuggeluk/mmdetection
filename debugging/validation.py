"""Validation.

Allows direct access to validation loops and allows debugging of those loops.
Note: This runs on CPU and not CUDA since we want to access the values and
manipulate them more easily

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
import numpy as np
import mmcv
from mmcv import Config, ProgressBar

from collections import OrderedDict

from mmdet.models.builder import build_backbone, build_neck, build_head, \
    build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmdet.datasets import build_dataset, build_dataloader
from mmdet.core import wrap_fp16_model

from argparse import ArgumentParser
from os.path import split

from PIL import Image


def parse_arguments():
    parser = ArgumentParser(description='debug the validation method')

    parser.add_argument('CONFIG', type=str, help='configuration file path')
    parser.add_argument('-v', '--validate', type=str,
                        help='sets to run validation on the network using '
                             'QualitAI metrics. Requires a path to the '
                             'checkpoint.')
    parser.add_argument('--out', type=str, help='output result file')

    return parser.parse_args()


class ValidationDebug:
    def __init__(self, config_path):
        """Initializes the network and dataset."""
        cfg = Config.fromfile(config_path)
        self.cfg = cfg

        # Now make the dataloader
        self.dataset = build_dataset(cfg.data.test)

        self.loader = build_dataloader(
            self.dataset,
            imgs_per_gpu=1,
            workers_per_gpu=0,
            dist=False,
            shuffle=False
        )

    def run(self):
        """Runs validation only with the network."""
        # Get the checkpoint file
        print('loading checkpoint file ...')
        cp = torch.load(self.cfg.work_dir + '/latest.pth')
        print('done')

        print('loading state dictionary ...')
        # Initialize network first as separate modules so we can access WFCOS
        backbone = build_backbone(self.cfg.model.backbone).cuda()
        neck = build_neck(self.cfg.model.neck).cuda()
        head = build_head(self.cfg.model.bbox_head).cuda()

        # Load the state dicts
        backbone_state = OrderedDict()
        neck_state = OrderedDict()
        head_state = OrderedDict()

        for key in cp['state_dict'].keys():
            if 'backbone' in key:
                backbone_state[key.split('.', 1)[1]] = cp['state_dict'][key]
            elif 'neck' in key:
                neck_state[key.split('.', 1)[1]] = cp['state_dict'][key]
            elif 'bbox_head' in key:
                head_state[key.split('.', 1)[1]] = cp['state_dict'][key]

        backbone.load_state_dict(backbone_state)
        neck.load_state_dict(neck_state)
        head.load_state_dict(head_state)

        # Set to eval mode
        backbone.eval()
        neck.eval()
        head.eval()

        print('done')

        print('starting inference validation run ...')
        for i, (img, cls) in enumerate(self.loader):
            out = self.backbone(img)
            out = self.neck(out)
            out = self.head(out)

            img_metas = [{'img_shape': (640, 800),
                          'scale_factor': 1}]
            bboxes = self.head.get_bboxes(out[0], out[1], out[2], img_metas,
                                          self.cfg.test_cfg)
            pass
        print('done')

    def validate(self, checkpoint_file_path, output_file):
        """Runs validation with QualitAI metrics."""

        print('Loading model...')
        self.cfg.data.test.test_mode = True
        self.cfg.model.pretrained = None

        model = build_detector(self.cfg.model, train_cfg=None,
                               test_cfg=self.cfg.test_cfg)
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, checkpoint_file_path,
                                     map_location='cpu')

        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = self.dataset.CLASSES

        model = MMDataParallel(model, device_ids=[0]).cuda()
        model.eval()
        print('Done!')
        print('Starting inference run...')

        # Do inference
        results = []
        bool_preds = []
        bool_targets = []
        prog_bar = ProgressBar(len(self.loader.dataset))

        file_id_lookup = self.get_ids_of_files(self.dataset.coco)

        for i, data in enumerate(self.loader):
            with torch.no_grad():
                result= model(return_loss=False, rescale=True, **data)
            results.append(result)

            img_shape = data['img_meta'][0].data[0][0]['ori_shape']
            bool_pred = self.transform_preds_to_boolean(
                    img_shape[0:2],
                    result[0]
                )
            bool_preds.append(bool_pred)
            img_name = split(data['img_meta'][0].data[0][0]['filename'])[1]
            img_id = file_id_lookup[img_name]

            bool_target = self.transform_targets_to_boolean(
                self.dataset.coco, img_id, img_shape[0:2])
            bool_targets.append(bool_target)

            target_img = np.zeros(img_shape, dtype='uint8')
            target_img[bool_target] = [0, 255, 0]
            target_img = Image.fromarray(target_img)
            pred_img = np.zeros(img_shape, dtype='uint8')
            pred_img[bool_pred] = [255, 0, 0]
            pred_img = Image.fromarray(pred_img)
            intersection_img = np.zeros(img_shape, dtype='uint8')
            intersection_img[bool_target * bool_pred] = [0, 0, 255]
            intersection_img = Image.fromarray(intersection_img)
            target_img.save('/workspace/outputs/{}-target.png'.format(i))
            pred_img.save('/workspace/outputs/{}-pred.png'.format(i))
            intersection_img.save('/workspace/outputs/{}-intersection.png'
                                  .format(i))

            prog_bar.update()

        # Calculate values
        print('\nStarting evaluation according to QualitAI metrics...')
        accuracy = 0.
        precision = 0.
        recall = 0.
        num_imgs = 0.
        for target, pred in zip(bool_targets, bool_preds):
            accuracy += self.calculate_accuracy(target, pred)
            precision += self.calculate_precision(target, pred)
            recall += self.calculate_recall(target, pred)
            num_imgs += 1.

        accuracy /= num_imgs
        precision /= num_imgs
        recall /= num_imgs

        print('Done!')

        print("\nResults:")
        print("======================")
        print("Num imgs:  {}".format(int(num_imgs)))
        print("Accuracy:  {:.7f}".format(accuracy))
        print("Precision: {:.7f}".format(precision))
        print("Recall:    {:.7f}".format(recall))

    @staticmethod
    def get_ids_of_files(dataset):
        """Gets file_name: img_id pairings of a datset.

        Args:
            dataset (pycocotools.COCO): The dataset.

        Returns:
            dict: A dict with file names as keys and img_ids as values.
        """
        out_dict = dict()
        for key in dataset.imgs.keys():
            out_dict[dataset.imgs[key]['file_name']] = key

        return out_dict

    @staticmethod
    def transform_targets_to_boolean(dataset, img_id, img_size):
        """Transforms targets to boolean values.

        Args:
            dataset (pycocotools.COCO): The coco dataset.
            img_id (int): The img_id of the current image


        Returns:
            np.ndarray: A (h, w) bool array where areas with targets are true.
        """
        ann_ids = dataset.getAnnIds(img_id)
        anns = dataset.loadAnns(ann_ids)
        out_mask = np.zeros(img_size, dtype=bool)
        adder = np.array([0., 1., 0., 1.])

        for ann in anns:
            bbox = (np.array(ann['bbox']) + adder).astype(int)
            bbox[3] = bbox[1] + bbox[3]
            bbox[2] = bbox[0] + bbox[2]
            out_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = True

        return out_mask

    @staticmethod
    def transform_preds_to_boolean(img_size, preds):
        """Transforms preds to boolean values.

        Args:
            img_size (tuple): The full image size in (h, w)
            preds (np.array): (n, 4) The predicted bounding boxes.

        Returns:
            np.ndarray: A (h, w) boolean numpy array where areas with
                detections are true.
        """
        out = np.zeros(img_size, dtype=bool)

        adder = np.array([0., 1., 0., 1.])

        for i in preds:
            bbox = (i[0:4] + adder).astype(int)
            out[bbox[1]:bbox[3], bbox[0]:bbox[2]] = True

        return out


    @staticmethod
    def calculate_accuracy(targets, preds):
        """Takes targets and preds and produces accuracy values.

        Shape:
            targets: (h, w) boolean tensor
            preds: (h, w) boolean tensor

        Args:
            targets (np.ndarray): A boolean tensor with the target values.
            preds (np.ndarray): A boolean tensor with the predicted values.

        Returns:
            float: The accuracy.
        """
        intersection_foreground = targets * preds
        intersection_background = np.invert(targets) * np.invert(preds)

        acc_foreground = float(np.sum(intersection_foreground)) \
                         / (float(np.sum(targets)) + 1e-7)
        acc_background = float(np.sum(intersection_background)) \
                         / (float(np.sum(np.invert(targets))) + 1e-7)
        return (acc_foreground + acc_background) / 2

    @staticmethod
    def calculate_precision(targets, preds):
        """Takes targets and preds and produces precision values.

        Shape:
            targets: (h, w) boolean tensor
            preds: (h, w) boolean tensor

        Args:
            targets (np.ndarray): A boolean tensor with the target values.
            preds (np.ndarray): A boolean tensor with the predicted values.

        Returns:
            float: The precision.
        """
        intersection_foreground = targets * preds
        n_intersection_foreground = float(np.sum(intersection_foreground))
        n_preds = float(np.sum(preds))

        return n_intersection_foreground / (n_preds + 1e-7)

    @staticmethod
    def calculate_recall(targets, preds):
        """Takes targets and preds and prodcues recall values.

        Shape:
            targets: (h, w) boolean tensor
            preds: (h, w) boolean tensor

        Args:
            targets (np.ndarray): A boolean tensor with the target values.
            preds (np.ndarray): A boolean tensor with the predicted values.

        Returns:
            float: The recall.
        """
        intersection_foreground = targets * preds
        n_intersection_foreground = float(np.sum(intersection_foreground))
        n_targets = float(np.sum(targets))

        return n_intersection_foreground / (n_targets + 1e-7)


if __name__ == '__main__':
    args = parse_arguments()
    vd = ValidationDebug(args.CONFIG)
    if args.validate:
        vd.validate(args.validate, args.out)
    else:
        vd.run()
