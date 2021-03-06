_base_ = [
    '../../_base_/datasets/dsV2_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
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
                num_channels=(32, 64, 64)),
            stage4=dict(
                num_modules=3,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 64)))),
    neck=dict(
        type='HRFPN_upsamp',
        in_channels=[32, 64, 64],
        out_channels=256,
        stride=1,
        num_outs=3),
    bbox_head=dict(
        type='WFCOSHead',
        num_classes=136,
        in_channels=256,
        max_energy=20,
        stacked_convs=4,
        feat_channels=256,
        strides=[1, 2, 4],
        regress_ranges=((-1, 9), (9, 20), (20, 1e8)),
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
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[16, 22])
checkpoint_config = dict(interval=100)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbVisualLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 1000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/wfcos_hrnet_ds_ext'
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
