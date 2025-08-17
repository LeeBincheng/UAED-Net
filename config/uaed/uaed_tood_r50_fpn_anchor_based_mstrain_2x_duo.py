_base_ = './uaed_tood_r50_fpn_2x_duo.py'
data_root = '/data/XXX/'
# learning policy
model = dict(bbox_head=dict(anchor_type='anchor_based'))
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadAugmentedImageFromFile',
         with_aug=True,
         aug_root=data_root + 'annotations/augmented/train/',
         to_float32=True,
         color_type='color'),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','aug_img']),
]
data = dict(train=dict(pipeline=train_pipeline))
