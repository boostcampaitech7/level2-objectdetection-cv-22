class Config22:
    # ───────────────────────────── 공통 설정 ─────────────────────────────
    def __init__(self):
        # 데이터셋 이름
        self.coco_dataset_train = 'coco_trash_train'
        self.coco_dataset_test = 'coco_trash_test'

        self.coco_fold_train = 'coco_fold_train'
        self.coco_fold_test = 'coco_fold_test'

        # 파일 경로 설정
        self.path_dataset = '/data/ephemeral/home/dataset/'
        self.path_output = '/data/ephemeral/home/outputs/'
        self.path_output_eval = self.path_output + 'output_eval/'

        # 파일 이름 설정
        self.filename_fold_train = 'train_fold_'
        self.filename_fold_val = 'val_fold_'
        self.filename_fold_output = 'fold_'
        self.filename_weights = 'model_final.pth'

        # Detectron2 및 모델 설정
        self.models = [
            "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            "data/ephemeral/home/Swin-Transformer-Object-Detection/config/faster_rcnn_r101_2x.py"
        ]
        self.path_model_pretrained = self.models[0]
        self.model_name = self.path_model_pretrained.split("/")[1].split(".")[0]

        # 기타 설정
        self.seed = 22
        self.NUM_WORKERS = 4
        self.SOLVER_IMS_PER_BATCH = 4

        # 백본 및 ROI 헤드 설정
        self.BACKBONE_NAME = ''
        self.ROI_HEADS_BATCH_SIZE_PER_IMAGE = 128  # Default : 128
        self.ROI_HEADS_NUM_CLASSES = 10
        self.RPN_NMS_THRESH = 0.7  # Default : 0.7

        # ───────────────────────────── 학습 설정 ─────────────────────────────
        # /data/ephemeral/home/repo/stratified_group_kfold/train_detectron_fixed.py 관련
        self.kfold = 5
        self.stratified = 1  # StratifiedGroupFold 라이브러리: 0, custom: 1

        self.EVAL_PERIOD = 500
        self.BASE_LR = 0.001
        self.MAX_ITER = 20000

        # ───────────────────────────── 추론 설정 ─────────────────────────────
        self.ensemble = 'mean'  # 평균: 'mean' / 다수결: 'major'
        self.ROI_HEADS_SCORE_THRESH_TEST = 0.05  # Default : 0.05
        self.ROI_HEADS_NMS_THRESH_TEST = 0.5  # Default : 0.5
