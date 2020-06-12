dataset_type = 'DeepScoresV2Dataset'
data_root = 'data/ds2_dense/'

img_norm_cfg = dict(
    mean=[240, 240, 240],
    std=[57, 57, 57],
    to_rgb=False)
# import numpy as np
#img_scale_train = np.asarray([200, 200])
#img_scale_test = np.asarray([200, 200])
#img_scale_test = np.asarray([3000, 3828])
#Base size = 2700 * 3828

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(2000, 3000)),
    dict(type='Resize', img_scale=tuple((1000, 1500)), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomCrop', crop_size=(300, 300)),
#     dict(type='Resize', img_scale=tuple((300, 300)), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1400, 1900),
        flip=False,
        transforms=[
#           dict(type='RandomCrop', crop_size=(200, 200)),
            dict(type='Resize',  keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomCrop', crop_size=(200, 200)),
#     dict(type='Resize', img_scale=tuple((200, 200)), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_oriented_train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_oriented_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_oriented_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')