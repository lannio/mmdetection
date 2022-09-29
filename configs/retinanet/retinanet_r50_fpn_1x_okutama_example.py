# model settings
model = dict(
    type='RetinaNet',    # 可以看出，RetinaNet 算法采用了 ResNet50 作为 Backbone, 并且考虑到整个目标检测网络比较大，前面部分网络没有进行训练，BN 也不会进行参数更新。需要说明的是上述默认配置是经过前人工作和 OpenMMLab 在 COCO 数据集上不断实践的结果。推荐大家直接使用该配置模式，效果相对比较稳定。
    backbone=dict(
        type='ResNet', # 骨架类名，后面的参数都是该类的初始化参数
        depth=50, # 表示使用 ResNet50
        num_stages=4, # ResNet 系列包括 stem+ 4个 stage 输出 1+3*3+3*4+3*6+3*3+1
        out_indices=(0, 1, 2, 3),  # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要， # 其 stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        frozen_stages=1, # 表示固定 stem 加上第一个 stage 的权重，不进行训练
        norm_cfg=dict(type='BN', requires_grad=True), # 所有的 BN 层的可学习参数都需要梯度
        norm_eval=True, # backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新
        style='pytorch', # 默认采用 pytorch 模式
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), # 使用 pytorch 提供的在 imagenet 上面训练过的权重作为预训练权重    
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], # 骨架多尺度特征图输出通道
        out_channels=256,  # 增强后通道输出 # FPN 输出的每个尺度输出特征图通道 # 说明了 5 个输出特征图的通道数都是 256
        start_level=1, # 从输入多尺度特征图的第几个开始计算 # 说明虽然输入是 4 个特征图，但是实际上 FPN 中仅仅用了后面三个
        add_extra_convs='on_input',  # 输出num_outs个多尺度特征图 # 额外输出层的特征图来源 #  说明额外输出的 2 个特征图的来源是骨架网络输出，而不是 FPN 层本身输出又作为后面层的输
        num_outs=5),  # FPN 输出特征图个数 #说明 FPN 模块虽然是接收 3 个特征图，但是输出 5 个特征图
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1, # COCO 数据集类别个数
        in_channels=256, # FPN 层输出特征图通道数
        stacked_convs=4, # 每个分支堆叠4层卷积
        feat_channels=256, # 中间特征图通道数
        anchor_generator=dict( #  anchor 生成过程
            type='AnchorGenerator',
            octave_base_scale=4, # 特征图 anchor 的 base scale, 值越大，所有 anchor 的尺度都会变大
            scales_per_octave=3, # 每个特征图有3个尺度，2**0, 2**(1/3), 2**(2/3)
            ratios=[0.5, 1.0, 2.0], # 每个特征图有3个高宽比例
            strides=[8, 16, 32, 64, 128]), # 特征图对应的 stride，必须特征图 stride 一致，不可以随意更改
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder', # RetinaNet 采用的编解码函数是主流的 DeltaXYWHBBoxCoder
            target_means=[.0, .0, .0, .0], # target_means 和 target_stds 相当于对 bbox 回归的 4 个 txtytwth 进行变换。在不考虑
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss', # 分类 loss
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)), # 回归 loss
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner', # 最大 IoU 原则分配器
            pos_iou_thr=0.5, # 正样本阈值
            neg_iou_thr=0.4, # 负样本阈值
            min_pos_iou=0,  # 正样本阈值下限
            ignore_iof_thr=-1), # 忽略 bboes 的阈值，-1表示不忽略
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

#--------------------------------------------
dataset_type = 'okutamaC1Dataset'
data_root = 'data/okutama_coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(720, 640), keep_ratio=True),
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
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_class1.json',
        img_prefix=data_root + 'train_crop/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_class1.json',
        img_prefix=data_root + 'test_crop/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_class1.json',
        img_prefix=data_root + 'test_crop/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

#--------------------------------------------
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

#--------------------------------------------
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


'''
^^^^backbone
https://zhuanlan.zhihu.com/p/353235794
(1) out_indices
ResNet 提出了骨架网络设计范式即 stem+n stage+ cls head，对于 ResNet 而言，其实际 forward 流程是 stem -> 4 个 stage -> 分类 head，stem 的输出 stride 是 4，而 4 个 stage 的输出 stride 是 4,8,16,32，这 4 个输出就对应 out_indices 索引。例如如果你想要输出 stride=4 的特征图，那么你可以设置 out_indices=(0,)，如果你想要输出 stride=4 和 8 的特征图，那么你可以设置 out_indices=(0, 1)。
因为 RetinaNet 后面需要接 FPN，故需要输出 4 个尺度特征图，简要代码如下：
for i, layer_name in enumerate(self.res_layers):
    res_layer = getattr(self, layer_name)
    x = res_layer(x)
    # 如果 i 在 self.out_indices 中才保留
    if i in self.out_indices:
        outs.append(x)
(2) frozen_stages
该参数表示你想冻结前几个 stages 的权重，ResNet 结构包括 stem+4 stage
frozen_stages=-1，表示全部可学习
frozen_stage=0，表示stem权重固定
frozen_stages=1，表示 stem 和第一个 stage 权重固定
frozen_stages=2，表示 stem 和前两个 stage 权重固定
(3) norm_cfg 和 norm_eval
norm_cfg 表示所采用的归一化算子，一般是 BN 或者 GN，而 requires_grad 表示该算子是否需要梯度，也就是是否进行参数更新，而布尔参数 norm_eval 是用于控制整个骨架网络的归一化算子是否需要变成 eval 模式。
(4) style
style='caffe' 和 style='pytorch' 的差别就在 Bottleneck 模块中
Bottleneck 是标准的 1x1-3x3-1x1 结构，考虑 stride=2 下采样的场景，caffe 模式下，stride 参数放置在第一个 1x1 卷积上，而 Pyorch 模式下，stride 放在第二个 3x3 卷积上：

^^^^neck
https://blog.csdn.net/baidu_36913330/article/details/119762293
将 c3、c4 和 c5 三个特征图全部经过各自 1x1 卷积进行通道变换得到 m3~m5，输出通道统一为 256
从 m5(特征图最小)开始，先进行 2 倍最近邻上采样，然后和 m4 进行 add 操作，得到新的 m4
将新 m4 进行 2 倍最近邻上采样，然后和 m3 进行 add 操作，得到新的 m3
对 m5 和新融合后的 m4、m3，都进行各自的 3x3 卷积，得到 3 个尺度的最终输出 P5～P3
将 c5 进行 3x3 且 stride=2 的卷积操作，得到 P6
将 P6 再一次进行 3x3 且 stride=2 的卷积操作，得到 P7
P6 和 P7 目的是提供一个大感受野强语义的特征图，有利于大物体和超大物体检测。 在 RetinaNet 的 FPN 模块中只包括卷积，不包括 BN 和 ReLU。
总结：FPN 模块接收 c3, c4, c5 三个特征图，输出 P3-P7 五个特征图，通道数都是 256, stride 为 (8,16,32,64,128)，其中大 stride (特征图小)用于检测大物体，小 stride (特征图大)用于检测小物体。


^^^^head 
Head
(1)论文中作者认为 one-stage 算法 head 设计比较关键，对最终性能影响较大，相比于其余 one-stage 算法，RetinaNet 的 Head 模块比较重量级，输出头包括分类和检测两个分支，且每个分支都包括 4 个卷积层，不进行参数共享，分类 Head 输出通道是 num_class*K，检测 head 输出通道是4*K, K 是 anchor 个数, 虽然每个 Head 的分类和回归分支权重不共享，但是 5 个输出特征图的 Head 模块权重是共享的。
(2)anchor generator
从上面配置可以看出：RetinaNet 一共 5 个输出特征图，每个特征图上有 3 种尺度和 3 种宽高比，每个位置一共 9 个 anchor，并且通过 octave_base_scale 参数来控制全局 anchor 的 base scales ，如果自定义数据集中普遍都是大物体或者小物体，则可能修改更改 octave_base_scale 参数。
gen_single_level_base_anchors
'''

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