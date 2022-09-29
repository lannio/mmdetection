_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/okutama_detection_c13.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(720, 640), keep_ratio=True, weather=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug', # 多尺度测试
        img_scale=(720, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, weather=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

model = dict(
    bbox_head=dict(
        num_classes=13, # COCO 数据集类别个数
    anchor_generator=dict( #  anchor 生成过程
        type='AnchorGenerator',
        octave_base_scale=4, # 特征图 anchor 的 base scale, 值越大，所有 anchor 的尺度都会变大
        scales_per_octave=3, # 每个特征图有3个尺度，2**0, 2**(1/3), 2**(2/3)
        ratios=[0.5, 1.0, 2.0], # 每个特征图有3个高宽比例
        strides=[8, 16, 32]), # 特征图对应的 stride，必须特征图 stride 一致，不可以随意更改
    ), # 回归 loss
    neck=dict(
        type='ATTN_3_FPN',
        in_channels=[256, 512, 1024, 2048], # 骨架多尺度特征图输出通道
        out_channels=256,  # 增强后通道输出 # FPN 输出的每个尺度输出特征图通道 # 说明了 5 个输出特征图的通道数都是 256
        start_level=1, # 从输入多尺度特征图的第几个开始计算 # 说明虽然输入是 4 个特征图，但是实际上 FPN 中仅仅用了后面三个
        add_extra_convs='on_input',  # 输出num_outs个多尺度特征图 # 额外输出层的特征图来源 #  说明额外输出的 2 个特征图的来源是骨架网络输出，而不是 FPN 层本身输出又作为后面层的输
        num_outs=5),
)


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

'''
retinanet 表示算法名称
r50 等表示骨架网络名
caffe 和 PyTorch 是指 Bottleneck 模块的区别，省略情况下表示是 PyTorch，后面会详细说明
fpn 表示 Neck 模块采用了 FPN 结构
mstrain 表示多尺度训练，一般对应的是 pipeline 中 Resize 类
1x 表示 1 倍数的 epoch 训练即 12 个 epoch，2x 则表示 24 个 epcoh
coco 表示在 COCO 数据集上训练
'''

# --> '../_base_/models/retinanet_r50_fpn.py'