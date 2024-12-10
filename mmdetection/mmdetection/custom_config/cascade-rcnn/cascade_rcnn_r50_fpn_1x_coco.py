_base_ = [
    '../../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/coco_detection_trash.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime_best.py'
]

name = 'cascade_rcnn_r50_fpn_1x_coco'
work_dir='./work_dirs/' + name

for i in range(len(_base_.model.roi_head.bbox_head)):
    _base_.model.roi_head.bbox_head[i].num_classes = 10

data_root = '/data/ephemeral/home/dataset/'
fp16 = dict(loss_scale='dynamic')
max_epochs = 12

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'k-fold/val_fold_3.json',
    metric='bbox',
    format_only=True,
    classwise=True,
    outfile_prefix=work_dir + name + '/val')  

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=True,
    classwise=True,
    outfile_prefix=work_dir + name + '/test')  

train_cfg = dict(max_epochs=max_epochs)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'Object-Detection',
            'entity': 'ljh19990825-naver',
            'group': 'cascade_rcnn',
            'name': 'name'
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')