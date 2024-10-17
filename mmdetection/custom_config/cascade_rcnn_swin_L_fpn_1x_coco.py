
_base_ = [
    '/data/ephemeral/home/baseline/mmdetection/configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '/data/ephemeral/home/baseline/mmdetection/configs/_base_/datasets/coco_detection_trash.py',
    '/data/ephemeral/home/baseline/mmdetection/configs/_base_/schedules/schedule_1x.py',
 '/data/ephemeral/home/baseline/mmdetection/configs/_base_/default_runtime.py'
]

work_dir='/data/ephemeral/home/baseline/mmdetection/work_dirs/cascade_rcnn_swin_L_custom'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth' 


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

for i in range(len(_base_.model.roi_head.bbox_head)):
    _base_.model.roi_head.bbox_head[i].num_classes = 10

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,  # Swin-Large에 맞게 embed_dims 수정
        depths=[2, 2, 18, 2],  # Swin-Large에 맞게 depths 수정
        num_heads=[6, 12, 24, 48],  # Swin-Large에 맞게 num_heads 수정
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

# test 제출 csv 훅
#custom_hooks = [
#    dict(type='SubmissionHook')
#]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'entity': 'ljh19990825-naver',
            'group': 'cascade_rcnn',
            'name' : 'cascade_rcnn_swin_L_fpn_1x_coco'
         })
]

fp16 = dict(loss_scale='dynamic')

checkpoint_config = dict(interval=1, save_best='auto', max_keep_ckpts=3)

# evaluation 설정, validation loss를 기준으로 가장 좋은 모델을 저장
evaluation = dict(
    interval=1, 
    metric='bbox',  # bbox로 평가할 경우. 다른 기준을 사용하고 싶다면 해당 metric을 수정
    save_best='auto')  # 'auto'는 가장 좋은 성능에 기반해 자동으로 저장