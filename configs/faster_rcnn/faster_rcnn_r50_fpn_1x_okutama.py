_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/okutama_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=13)))
CLASSES = ('Other', 'Handshaking', 'Hugging', 'Reading', 'Drinking', 'Pushing/Pulling', 'Carrying',
           'Calling', 'Running', 'Walking', 'Lying', 'Sitting', 'Standing')