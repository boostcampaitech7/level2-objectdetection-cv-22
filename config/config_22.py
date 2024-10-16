class Config22:
    # ──────────────────────────────── 공통 설정 ───────────────────────────────────
    # 데이터셋 이름
    coco_dataset_train = 'coco_trash_train'
    coco_dataset_test = 'coco_trash_test'

    coco_fold_train = 'coco_fold_train'
    coco_fold_test = 'coco_fold_test'

    # 파일 경로
    path_dataset = '/data/ephemeral/home/dataset/'

    path_output = '/data/ephemeral/home/outputs/'
    path_output_eval = path_output + 'output_eval/'

    filename_fold_train = 'train_fold_'
    filename_fold_val = 'val_fold_'
    filename_fold_output = 'output_fold_'
    filename_weights = 'model_final.pth'

    ############################################################
    # 추론 시 폴더 이름 반드시 넣어주기
    filename_this = 'faster_rcnn_R_101_FPN_3x_10-16 15:17output_fold_3'

    path_model_pretrained = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    #path_model_pretrained = "data/ephemeral/home/Swin-Transformer-Object-Detection/config/faster_rcnn_r101_2x.py"

    # 기타
    seed = 22
    model_name = path_model_pretrained.split("/")[1].split(".")[0]

    NUM_WOREKRS = 4
    BACKBONE_NAME = ''

    ROI_HEADS_BATCH_SIZE_PER_IMAGE = 128                # Default : 128
    ROI_HEADS_NUM_CLASSES = 10
    ROI_HEADS_SCORE_THRESH_TEST = 0.05                   # Default : 0.05
    ROI_HEADS_NMS_THRESH_TEST = 0.6                     # Default : 0.5

    RPN_NMS_THRESH = 0.5                                # Default : 0.7
    
    # ──────────────────────────────── 학습 설정 ───────────────────────────────────

    # StratifiedGroupFold 라이브러리 : 0, custom : 1
    kfold = 5
    stratified = 1

    # ──────────────────────────────── 추론 설정 ───────────────────────────────────

    ensemble = 'mean'                                     # 평균: mean / 다수결 : major