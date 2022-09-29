_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
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