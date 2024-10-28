import os
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader

# 개인화
# export PYTHONPATH=$PYTHONPATH:/data/ephemeral/home/repo/stratified_group_kfold/inference
from config.config_22 import Config22
from inference_ensemble import ensemble_predictions
from inference_utils import MyMapper

"""
    stratifiedGroupKFold로 나눈 파일 전체를 평균 앙상블하여 inference
"""
# 경로 설정 ─────────────────────────────────────────────────────────────────────────────────

k = Config22.kfold
seed = Config22.seed

coco_dataset_test = Config22.coco_dataset_test
coco_fold_test = Config22.coco_fold_test

path_dataset = Config22.path_dataset

path_output = Config22.path_output
path_output_eval = Config22.path_output_eval

path_model_pretrained = Config22.path_model_pretrained

filename_fold_train = Config22.filename_fold_train
filename_fold_val = Config22.filename_fold_val
filename_fold_output = Config22.filename_fold_output
filename_weights = Config22.filename_weights
filename_this = Config22.filename_this

ensemble = Config22.ensemble

# ───────────────────────────────────────────────────────────────────────────────────────────

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(path_model_pretrained))
    cfg.OUTPUT_DIR = path_output + filename_fold_output + filename_this
    cfg.DATASETS.TEST = (coco_dataset_test,)

    cfg.DATALOADER.NUM_WOREKRS = Config22.NUM_WOREKRS
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = Config22.ROI_HEADS_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = Config22.ROI_HEADS_NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Config22.ROI_HEADS_SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = Config22.ROI_HEADS_NMS_THRESH_TEST

    cfg.MODEL.RPN.NMS_THRESH = Config22.RPN_NMS_THRESH

    return cfg


all_targets = []

prediction_strings = []
file_names = []


if coco_dataset_test not in DatasetCatalog.list():
    register_coco_instances(coco_dataset_test, {}, path_dataset + 'test.json', path_dataset)

cfg = setup_cfg()
test_loader = build_detection_test_loader(cfg, coco_dataset_test, MyMapper)

for data in tqdm(test_loader):
    
    prediction_string = ''
    fold_outputs = []
    data = data[0]
    
    for fold_idx in range(5):
        path_weight = os.path.join(f'{cfg.OUTPUT_DIR}{fold_idx}', filename_weights)
        
        if not os.path.exists(path_weight):    
            print("파일 없음", path_weight)
            continue
        
        cfg.MODEL.WEIGHTS = path_weight
        predictor = DefaultPredictor(cfg)

        outputs = predictor(data['image'])['instances']

        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()

        fold_outputs.append({
            'targets': outputs.pred_classes.cpu().tolist(),
            'boxes': [i.cpu().detach().numpy() for i in outputs.pred_boxes],
            'scores': outputs.scores.cpu().tolist(),
        })

    
    targets, boxes, scores = ensemble_predictions(fold_outputs, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    
    for target, box, score in zip(targets,boxes,scores):
        prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
        + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')


    prediction_strings.append(prediction_string)

    file_name = os.path.basename(data['file_name'])
    file_names.append(data['file_name'])


# submission 파일 저장
submission = pd.DataFrame({
    'PredictionString': prediction_strings,
    'image_id': file_names
})
submission.to_csv(os.path.join(path_output + filename_this, f'submission_{filename_this}.csv'), index=False)