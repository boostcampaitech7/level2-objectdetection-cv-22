import os
from tqdm import tqdm
import pandas as pd
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader

# 시각화
from detectron2.utils.visualizer import Visualizer
import cv2

import sys
sys.path.append('/data/ephemeral/home/repo')

# 개인화
from config.config_22 import Config22

from stratified_group_kfold.inference.inference_utils import MyMapper

"""
    고정된 1개의 파일만 inference
    (전체 파일에 대한 시각화 포함)
"""

# 경로 설정 ─────────────────────────────────────────────────────────────────────────────────

k = Config22.kfold
seed = Config22.seed
visualized = Config22.visualized

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

path_weight = os.path.join(f'{path_output}{filename_this}', filename_weights)

# ───────────────────────────────────────────────────────────────────────────────────────────

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(path_model_pretrained))
    cfg.OUTPUT_DIR = path_output + filename_this
    cfg.DATASETS.TEST = (coco_dataset_test,)
    
    cfg.MODEL.WEIGHTS = path_weight

    cfg.DATALOADER.NUM_WOREKRS = Config22.NUM_WOREKRS
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = Config22.ROI_HEADS_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = Config22.ROI_HEADS_NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Config22.ROI_HEADS_SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = Config22.ROI_HEADS_NMS_THRESH_TEST

    cfg.MODEL.RPN.NMS_THRESH = Config22.RPN_NMS_THRESH

    return cfg


prediction_strings = []
file_names = []


if coco_dataset_test not in DatasetCatalog.list():
    register_coco_instances(coco_dataset_test, {}, path_dataset + 'test.json', path_dataset)

cfg = setup_cfg()
test_loader = build_detection_test_loader(cfg, coco_dataset_test, MyMapper, num_workers=4)
predictor = DefaultPredictor(cfg)

for data in tqdm(test_loader):
    
    prediction_string = ''
    
    data = data[0]
    
    outputs = predictor(data['image'])['instances']
    
    targets = outputs.pred_classes.cpu().tolist()
    boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
    scores = outputs.scores.cpu().tolist()
    
    for target, box, score in zip(targets,boxes,scores):
        prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
        + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
    
    prediction_strings.append(prediction_string)
    file_names.append(data['file_name'].replace(path_dataset,''))

    # 시각화 코드
    if visualized:
        v = Visualizer(data['image'], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs.to("cpu"))

        result_file = os.path.join(cfg.OUTPUT_DIR, f"visualization_{data['file_name'].split('/')[-1]}")
        cv2.imwrite(result_file, out.get_image()[:, :, ::-1])
 

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det_{filename_this}.csv'), index=False)