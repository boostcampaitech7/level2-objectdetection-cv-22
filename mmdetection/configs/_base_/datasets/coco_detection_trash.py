
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset/'

metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(900, 900), keep_ratio=True), # 1024 -> 512 gpu 메모리 부족
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(900, 900), keep_ratio=True), # 1024 -> 512 gpu 메모리 부족
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/data/ephemeral/home/dataset/k-fold/train_fold_3.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=5,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/data/ephemeral/home/dataset/k-fold/val_fold_3.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=5,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/data/ephemeral/home/dataset/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))


val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/dataset/k-fold/val_fold_3.json',
    metric='bbox',
    format_only=True,
    classwise=True,
    outfile_prefix='./work_dirs/cascade_rcnn_r50_fpn_1x_coco/val')  # 이 줄 추가

# test_evaluator 수정
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/ephemeral/home/dataset/test.json',
    metric='bbox',
    format_only=True,
    classwise=True,
    outfile_prefix='./work_dirs/cascade_rcnn_r50_fpn_1x_coco/test')  # 이 줄 추가
