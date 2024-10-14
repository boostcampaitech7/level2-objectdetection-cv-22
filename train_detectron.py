import os
import numpy as np
import json

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from sklearn.model_selection import StratifiedGroupKFold
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from stratified_kfold import stratified_group_k_fold

from detectron2.utils.logger import setup_logger
setup_logger()

# def get_distribution(y):
#     y_distr = Counter(y)
#     y_vals_sum = sum(y_distr.values())

#     return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]


# distrs = [get_distribution(y)]
# index = ['training set']



# 경로 설정 ─────────────────────────────────────────────────────────────────────────────────

coco_fold_train = 'coco_fold_train'
coco_fold_test = 'coco_fold_test'

path_annotation = '/data/ephemeral/home/dataset'
path_dataset = '/data/ephemeral/home/dataset/'
path_model_pretrained = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

# ───────────────────────────────────────────────────────────────────────────────────────────


# COCO 데이터셋 등록 
def register_datasets(annotation_path, fold_idx, path):
    train_json = os.path.join(path_dataset, f'train_fold_{fold_idx}.json')
    val_json = os.path.join(path_dataset, f'val_fold_{fold_idx}.json')
    
    train_dataset_name = f'{coco_fold_train}{fold_idx}'
    val_dataset_name = f'{coco_fold_test}{fold_idx}'
    
    register_coco_instances(train_dataset_name, {}, train_json, path)
    register_coco_instances(val_dataset_name, {}, val_json, path)
    
    return train_dataset_name, val_dataset_name



def kfold_training(k, annotation_file, cfg, path_dataset):

    with open(annotation_file + '/train.json') as f: 
        data = json.load(f)

    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

    X = np.ones((len(data['annotations']),1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    #cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)
    cv = stratified_group_k_fold(X, y, groups, k)

    for fold_idx, (train_idx, val_idx) in enumerate(cv):

        train_name, val_name = register_datasets(annotation_file, fold_idx, path_dataset)
        
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = (val_name,)
        cfg.OUTPUT_DIR = f'./output_fold_{fold_idx}'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        

        train_model(cfg)


def train_model(cfg):
    trainer = DefaultTrainer(cfg)
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


kfold_training(5, path_annotation, cfg, path_dataset)