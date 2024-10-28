_base_ = [
    '../../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/coco_detection_trash.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime_best.py'
]

name = 'cascade_rcnn_r50_fpn_1x_coco'
work_dir='./work_dirs/' + name

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth' 

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,  
        depths=[2, 2, 18, 2],  
        num_heads=[6, 12, 24, 48], 
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536])) 

for i in range(len(_base_.model.roi_head.bbox_head)):
    _base_.model.roi_head.bbox_head[i].num_classes = 10

dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset/'
fp16 = dict(loss_scale='dynamic')
backend_args = None
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

optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'Object-Detection',
            'entity': 'ljh19990825-naver',
            'group': 'cascade_rcnn_fpn_1x_coco',
            'name': name
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')