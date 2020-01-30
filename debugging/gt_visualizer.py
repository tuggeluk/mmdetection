"""Ground Truth Visualizer.

Visualizes ground truth data locally, i.e. not to tensorboard.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 19, 2019
"""
from argparse import ArgumentParser
from torchvision.datasets.coco import CocoDetection
from pycocotools import mask
from PIL import Image, ImageFont
from debugging.constants import *
import numpy as np

import wandb


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description='visualizes ground truth images and'
                                        'annotations')

    parser.add_argument('IMAGES_ROOT', type=str,
                        help='path to directory containing the images')

    parser.add_argument('ANNOTATIONS', type=str,
                        help='path to the annotations file')

    parser.add_argument('-w', action='store_true',
                        help='runs with weights and biases')

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def extract_seg_masks(annotations: list) -> list:
    """Extract polygon from an annotation of an image."""
    polygons = []

    for ann in annotations:
        h, w = ann['bbox'][3::-2]
        polygons.append(mask.decode(mask.frPyObjects(ann['segmentation'],
                                                     h, w)))

    return polygons


def get_bboxes(annotations: list) -> list:
    """Get a list of all bounding boxes in an image."""
    return [ann['bbox'] for ann in annotations]


def get_labels(annotations: list) -> list:
    """Get the class label of an annotated object."""
    return [DEEP_SCORES_CLASSES[int(ann['category_id'])] for ann in annotations]


def draw_objects(objects: list, image: Image.Image, show: bool) -> Image.Image:
    """Draw objects on image.

    Args:
        objects: The objects to be drawn. Expected to be a list of tuples with
            the shape (polygon, bbox).
        image: The PIL image to be drawn.

    Returns:
        The drawn PIL image.
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size
    font = ImageFont.load_default()

    for seg_mask, bbox, label in objects:
        # bbox is relative to picture size (i.e. in range from 0 to 1) so we
        # have to multiply it to get the pixel positions. We do that here.
        px_bbox = [bbox[0] * w,
                   bbox[1] * h,
                   (bbox[0] + bbox[2]) * w,
                   (bbox[1] + bbox[3]) * h
                   ]
        # Draw bounding box
        draw.rectangle(
            px_bbox,
            outline='#ff0000',
            width=2
        )

        # Draw class label
        text_size = draw.textsize(label, font)
        draw.rectangle([px_bbox[0], px_bbox[1],
                        px_bbox[0] + 4 + text_size[0],
                        px_bbox[1] - 4 - text_size[1]],
                       fill=(0, 0, 0))

        draw.text([px_bbox[0] + 2, px_bbox[1] - text_size[1] - 2],
                  label, fill=(255, 255, 255),
                  font=font)

        # Draw seg masks
        # Turn seg mask into a PIL Image
        seg_mask *= 255
        try:
            grayscale = seg_mask.shape[2] == 1
        except IndexError:
            grayscale = True
        if grayscale:
            seg_mask = Image.fromarray(np.squeeze(seg_mask), mode='L')
        else:
            seg_mask = Image.fromarray(seg_mask)
        draw.bitmap(px_bbox[0:2], seg_mask, fill='#00ff7f')

    return image


if __name__ == '__main__':
    arguments = parse_args()

    wandb.init(project='debugging')
    wandb.config.lr = 0.01


    dataset = CocoDetection(arguments.IMAGES_ROOT, arguments.ANNOTATIONS)

    ds_length = len(dataset)
    for count, (img, ann) in enumerate(dataset):
        print("Showing image {} of {}".format(count, ds_length))

        # Extract all objects from the annotations into a polygon and bbox list
        objects = [i for i in zip(extract_seg_masks(ann), get_bboxes(ann),
                                  get_labels(ann))]

        drawn = draw_objects(objects, img, not arguments.w)

        if arguments.w:
            wandb.log({'img': wandb.Image(img)}, step=count)
            wandb.log({'gt': wandb.Image(drawn)}, step=count)
            wandb.log({'img_jpg': [wandb.Image(img)]}, step=count)
            wandb.log({'gt_jpg': [wandb.Image(img)]}, step=count)
            wandb.log({'combined': [wandb.Image(img), wandb.Image(drawn)]},
                      step=count)
        response = input('Show next image? q to exit\n')

        if response == 'q':
            print('Clearing memory...')
            break
