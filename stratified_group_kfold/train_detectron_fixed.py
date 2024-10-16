import os
import numpy as np
import json
from detectron2.utils.logger import setup_logger
setup_logger()

# 학습용
from custom_trainer import MyTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg

# 데이터 로딩
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# 개인화
# config경로 오류 시 터미널에서 ▼
# export PYTHONPATH=$PYTHONPATH:/data/ephemeral/home/repo/stratified_group_kfold
import sys
sys.path.append('/data/ephemeral/home/repo')
from config.config_22 import Config22
from datetime import datetime
import wandb



# 경로 설정 ─────────────────────────────────────────────────────────────────────────────────

k = Config22.kfold
seed = Config22.seed

title = f'{Config22.model_name}_{datetime.now().strftime("%m-%d %H:%M")}'

coco_dataset_train = Config22.coco_dataset_train
coco_dataset_test = Config22.coco_dataset_test

coco_fold_train = Config22.coco_fold_train
coco_fold_test = Config22.coco_fold_test

path_dataset = Config22.path_dataset

path_output = Config22.path_output
path_output_this = path_output + title

path_model_pretrained = Config22.path_model_pretrained

filename_fold_train = Config22.filename_fold_train
filename_fold_val = Config22.filename_fold_val
filename_fold_output = Config22.filename_fold_output

fold_idx = 3

os.makedirs(path_output, exist_ok=True)
wandb.init(project="2024 부스트캠프 재활용품 분류대회(22, CSV)", 
           name=title)

# ───────────────────────────────────────────────────────────────────────────────────────────


# COCO 데이터셋 등록 
def register_datasets(path_dataset, fold_idx):

    train_dataset_name = f'{coco_fold_train}_{fold_idx}'
    val_dataset_name = f'{coco_fold_test}_{fold_idx}'

    train_json = os.path.join(path_dataset, f'{filename_fold_train}{fold_idx}.json')
    val_json = os.path.join(path_dataset, f'{filename_fold_val}{fold_idx}.json')
    
    if train_dataset_name not in DatasetCatalog.list():
        register_coco_instances(train_dataset_name, {}, train_json, path_dataset)

    if val_dataset_name not in DatasetCatalog.list():    
        register_coco_instances(val_dataset_name, {}, val_json, path_dataset)

    data_size = len(DatasetCatalog.get(train_dataset_name))
    MetadataCatalog.get(train_dataset_name).thing_classes=['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                                                           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    
    return train_dataset_name, val_dataset_name, data_size



def train_model(cfg, path_dataset, fold_idx):

    train_name, val_name, data_size = register_datasets(path_dataset, fold_idx)
        
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.OUTPUT_DIR = f'{path_output_this}{filename_fold_output}{fold_idx}'

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()


# Config 설정
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(path_model_pretrained))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path_model_pretrained)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.TEST.EVAL_PERIOD = 500

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 8000
cfg.SOLVER.STEPS = (cfg.SOLVER.MAX_ITER // 2, 
                    cfg.SOLVER.MAX_ITER * 2 //3)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.DEVICE = "cuda"
cfg.SOLVER.AMP.ENABLED = True

# cfg.MODEL.ROI_HEADS.LOSS_WEIGHT_CLS = 2.0
#cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
# cfg.MODEL.RPN.NMS_THRESH = 0.5

# cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
# cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]


#cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5

register_datasets(path_dataset, fold_idx)
train_model(cfg, path_dataset, fold_idx)