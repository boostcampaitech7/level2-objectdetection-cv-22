_base_ = [
    '/data/ephemeral/home/baseline/mmdetection/configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '/data/ephemeral/home/baseline/mmdetection/configs/_base_/datasets/coco_detection_trash.py',
    '/data/ephemeral/home/baseline/mmdetection/configs/_base_/schedules/schedule_1x.py',
 '/data/ephemeral/home/baseline/mmdetection/configs/_base_/default_runtime.py'
]

work_dir='/data/ephemeral/home/baseline/mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x_coco'


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)

for i in range(len(_base_.model.roi_head.bbox_head)):
    _base_.model.roi_head.bbox_head[i].num_classes = 10

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


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'entity': 'ljh19990825-naver',
            'group': 'cascade_rcnn',
            'name' : 'cascade_rcnn_r50_fpn_1x_coco'
         })
]


wandb_kwargs = dict(
    project='mmdetection',
    entity='ljh19990825-naver',
    group='cascade_rcnn',
    name='cascade_rcnn_r50_fpn_1x_coco'
)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs=wandb_kwargs,
            interval=50,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100
        )
    ]
)



fp16 = dict(loss_scale='dynamic')

checkpoint_config = dict(interval=1, save_best='auto', max_keep_ckpts=3)

# evaluation 설정, validation loss를 기준으로 가장 좋은 모델을 저장
evaluation = dict(
    interval=1, 
    metric='bbox',  # bbox로 평가할 경우. 다른 기준을 사용하고 싶다면 해당 metric을 수정
    save_best='auto')  # 'auto'는 가장 좋은 성능에 기반해 자동으로 저장



