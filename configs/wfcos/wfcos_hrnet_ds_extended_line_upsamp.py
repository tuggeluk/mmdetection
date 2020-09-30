# model settings
model = dict(
    type='WFCOS',
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        type='HRNet_upsamp',
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
        num_classes=124,
        in_channels=256,
        max_energy=1,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            loss_weight=0.5),
        # loss_energy=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=5.0,
        #     loss_weight=10.0,
        #     reduction='sum'
        # ),
        loss_energy=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        split_convs=False,
        r=1.,
        bbox_percent=1.
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
# dataset settings
dataset_type = 'DeepScoresDataset'
data_root = 'data/deep_scores_dense_extended_lineinfo/'
img_norm_cfg = dict(
    mean=[240.15232515949037, 240.15229097456378, 240.15232515949037],
    std=[57.178083212078896, 57.178143244444556, 57.178083212078896],
    to_rgb=False)
img_scale_train = (800, 2000)
img_scale_test = (3000, 3828)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(600, 600)),
    dict(type='Resize', img_scale=img_scale_train, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_test,
        flip=False,
        transforms=[
            # dict(type='RandomCrop', crop_size=(640, 800)),
            dict(type='Resize', img_scale=img_scale_test, keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_train.json',
        img_prefix=data_root + 'images_png/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_val.json',
        img_prefix=data_root + 'images_png/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_val.json',
        img_prefix=data_root + 'images_png/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
     grad_clip=dict(
         max_norm=5.
     ))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[16, 22])
checkpoint_config = dict(interval=50)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        #dict(type='WandbLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 1000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/wfcos_hrnet_ds_extended_line_upsamp/'
load_from = None
resume_from = None
# resume_from = work_dir + '/epoch_4.pth'
workflow = [('train', 1)]

# wandb settings
wandb_cfg = dict(
    entity='warp-net',
    project='deepscores',
    dryrun=False
)
