# model settings
model = dict(
    type='WFCOS',
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        stride=2,
        num_outs=5),
    bbox_head=dict(
        type='WFCOSHead',
        num_classes=136,
        in_channels=256,
        max_energy=20,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 9), (9, 20), (20, 128), (128, 256), (256, 1e8)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.),
        loss_bbox=dict(
            type='IoULoss',
            loss_weight=1.0
        ),
        loss_energy=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            loss_weight=1.
        ),
        split_convs=False,
        assign="min_edge",
        r=5.
    ))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(type='nms', iou_thr=0.2),
    max_per_img=1000)

optimizer = dict(
    type='SGD',
    lr=0.0005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict( grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=3000,
    warmup_ratio=1.0/4,
    gamma=0.5,
    step=[80, 160])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbVisualLoggerHook')
    ])
# specific dataset configs
dataset_type = 'DeepScoresV2Dataset'
data_root = 'data/ds2_dense/'

img_norm_cfg = dict(
    mean=[240, 240, 240],
    std=[57, 57, 57],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=tuple((2700, 3828)), keep_ratio=True),
    #dict(type='Resize', img_scale=tuple((1400, 1920)), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(600, 1000)),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2700, 3828),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=tuple((2700, 3828)), keep_ratio=True),
            #dict(type='Resize', img_scale=(1400, 1920), keep_ratio=True),
            #dict(type='RandomCrop', crop_size=(800, 800)),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

evaluation = dict(interval=1000, metric='bbox')

# yapf:enable
# runtime settings
total_epochs = 1000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dsv2_wfcos_hrnet_normal'
load_from = None
# load_from = work_dir + '/epoch_4.pth'
resume_from = None
# resume_from = work_dir + '/epoch_4.pth'
workflow = [('train', 1)]

# wandb settings
wandb_cfg = dict(
    entity='warp-net',
    project='fcos-wfcos-baseline-ds_ext',
    dryrun=False
)
