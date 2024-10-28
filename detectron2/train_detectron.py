import os
import numpy as np
import json
from detectron2.utils.logger import setup_logger
setup_logger()

# 학습용
from custom_trainer import MyTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from sklearn.model_selection import StratifiedGroupKFold

# 데이터 로딩
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from stratified_kfold import stratified_group_k_fold, save_fold_data

import sys
sys.path.append('/data/ephemeral/home/repo')

# 개인화
from config.config_22 import Config22
from datetime import datetime
import wandb


# 경로 설정 ─────────────────────────────────────────────────────────────────────────────────

k = Config22.kfold
seed = Config22.seed

title = f'{Config22.model_name}_{datetime.now().strftime("%m-%d %H:%M")}'

coco_fold_train = Config22.coco_fold_train
coco_fold_test = Config22.coco_fold_test

path_dataset = Config22.path_dataset

path_output = Config22.path_output
path_output_this = path_output + title

path_model_pretrained = Config22.path_model_pretrained

filename_fold_train = Config22.filename_fold_train
filename_fold_val = Config22.filename_fold_val
filename_fold_output = Config22.filename_fold_output

os.makedirs(path_output, exist_ok=True)
wandb.init(project="2024 부스트캠프 재활용품 분류대회(22, CSV)", 
           name=title)

# ───────────────────────────────────────────────────────────────────────────────────────────


# COCO 데이터셋 등록 
def register_datasets(path_dataset, fold_idx):

    train_dataset_name = f'{coco_fold_train}{fold_idx}'
    val_dataset_name = f'{coco_fold_test}{fold_idx}'

    train_json = os.path.join(path_dataset, f'{filename_fold_train}{fold_idx}.json')
    val_json = os.path.join(path_dataset, f'{filename_fold_val}{fold_idx}.json')
    
    register_coco_instances(train_dataset_name, {}, train_json, path_dataset)
    register_coco_instances(val_dataset_name, {}, val_json, path_dataset)

    data_size = len(DatasetCatalog.get(train_dataset_name))
    MetadataCatalog.get(train_dataset_name).thing_classes=['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                                                           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    
    return train_dataset_name, val_dataset_name



def kfold_training(k, cfg, path_dataset):

    # 설정 경로 : /data/ephemeral/home/dataset/train.json
    with open(path_dataset + 'train.json') as f: 
        data = json.load(f)

    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

    X = np.ones((len(data['annotations']), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    if Config22.stratified == 0:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    elif Config22.stratified == 1:
        cv = stratified_group_k_fold(X, y, groups, k)


    for fold_idx, (train_idx, val_idx) in enumerate(cv):
        print(f"Training fold {fold_idx + 1}/{k}...")

        save_fold_data(data, train_idx, val_idx, fold_idx, path_dataset)

        train_name, val_name = register_datasets(path_dataset, fold_idx)
        
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = (val_name,)
        cfg.OUTPUT_DIR = f'{path_output_this}{filename_fold_output}{fold_idx}'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        train_model(cfg)


def train_model(cfg):
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


# Config 설정
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(path_model_pretrained))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path_model_pretrained)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = Config22.ROI_HEADS_NUM_CLASSES
cfg.TEST.EVAL_PERIOD = Config22.EVAL_PERIOD

cfg.SOLVER.IMS_PER_BATCH = Config22.SOLVER_IMS_PER_BATCH
cfg.SOLVER.BASE_LR = Config22.BASE_LR
cfg.SOLVER.MAX_ITER = Config22.MAX_ITER
cfg.SOLVER.STEPS = (cfg.SOLVER.MAX_ITER // 2, 
                    cfg.SOLVER.MAX_ITER * 2 //3)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = Config22.ROI_HEADS_BATCH_SIZE_PER_IMAGE
cfg.MODEL.DEVICE = "cuda"
cfg.SOLVER.AMP.ENABLED = True

# cfg.MODEL.ROI_HEADS.LOSS_WEIGHT_CLS = 2.0
#cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
# cfg.MODEL.RPN.NMS_THRESH = 0.5

# cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
# cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]


#cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5


kfold_training(k, cfg, path_dataset)