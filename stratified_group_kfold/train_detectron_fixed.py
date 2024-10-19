import os
from detectron2.utils.logger import setup_logger
setup_logger()

# 학습용
from detectron2 import model_zoo
from detectron2.config import get_cfg
from stratified_group_kfold.custom_trainer import MyTrainer

# 데이터 로딩
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from datetime import datetime
import wandb

""" 
    하나의 고정된 데이터셋만 사용하여 학습
    
    train 데이터 원본 사용 시: fold_idx = -1
    다른 데이터 사용 시 fold 번호 : fold_idx >= 0
"""

def log_executions(path, filename_output):
    with open(f"{path}record.txt", "a") as f:
        f.write(f"{filename_output}\n")

def choose_dataset(con22, fold_idx):
    if fold_idx < 0:
        t_name = con22.coco_dataset_train
        v_name = con22.coco_dataset_test
        t_json = os.path.join(con22.path_dataset, 'train.json')
        v_json = os.path.join(con22.path_dataset, 'test.json')
    else:
        t_name = f'{con22.coco_fold_train}_{fold_idx}'
        v_name = f'{con22.coco_fold_test}_{fold_idx}'
        t_json = os.path.join(con22.path_dataset, f'{con22.filename_fold_train}{fold_idx}.json')
        v_json = os.path.join(con22.path_dataset, f'{con22.filename_fold_val}{fold_idx}.json')

    return t_name, v_name, t_json, v_json


def register_datasets(con22, fold_idx):
    
    train_dataset_name, val_dataset_name, train_json, val_json = choose_dataset(con22, fold_idx)

    if train_dataset_name not in DatasetCatalog.list():
        register_coco_instances(train_dataset_name, {}, train_json, con22.path_dataset)

    if val_dataset_name not in DatasetCatalog.list():    
        register_coco_instances(val_dataset_name, {}, val_json, con22.path_dataset)

    data_size = len(DatasetCatalog.get(train_dataset_name))
    MetadataCatalog.get(train_dataset_name).thing_classes=['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                                                           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    
    return train_dataset_name, val_dataset_name, data_size



# Config 설정

def set_cfg(con22):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(con22.path_model_pretrained))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(con22.path_model_pretrained)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = con22.ROI_HEADS_NUM_CLASSES
    cfg.TEST.EVAL_PERIOD = con22.EVAL_PERIOD

    cfg.SOLVER.IMS_PER_BATCH = con22.SOLVER_IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = con22.BASE_LR
    cfg.SOLVER.MAX_ITER = con22.MAX_ITER
    cfg.SOLVER.STEPS = (cfg.SOLVER.MAX_ITER // 2, 
                        cfg.SOLVER.MAX_ITER * 2 //3)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = con22.ROI_HEADS_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.DEVICE = "cuda"
    cfg.SOLVER.AMP.ENABLED = True

    # cfg.MODEL.ROI_HEADS.LOSS_WEIGHT_CLS = 2.0
    # cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
    # cfg.MODEL.RPN.NMS_THRESH = 0.5

    # cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    # cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]


    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    # cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5

    return cfg


# train 설정

def train_model(con22, fold_idx=-1):
    
    train_name, val_name, data_size = register_datasets(con22, fold_idx)

    cfg = set_cfg(con22)
    title = f'[{datetime.now().strftime("%m/%d-%H%M")}]{con22.model_name}'
    path_output_this = con22.path_output + title
    log_executions(con22.path_output, title)

    cfg.OUTPUT_DIR = f'{path_output_this}{con22.filename_fold_output}{fold_idx}'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    wandb.init(project="2024 부스트캠프 재활용품 분류대회(22, CSV)", name=title)
        
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)

    trainer = MyTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()