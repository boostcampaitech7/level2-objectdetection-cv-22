#!/usr/bin/env python
# coding: utf-8

# ## fold 추론용

# ### 1. 모듈 불러오기

# In[1]:


import os
import copy
from tqdm import tqdm
import pandas as pd
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader

# fold 관련
import numpy as np


# ### 2. 데이터 등록
# #### 리소스 등록 및 경로 설정

# In[2]:


coco_dataset_test = 'coco_trash_test'
coco_fold_test = 'coco_fold_test'

path_dataset = '/data/ephemeral/home/dataset/'
path_output_eval = '/data/ephemeral/home/baseline/custom/output_eval'
path_output = '/data/ephemeral/home/baseline/custom/output_fold'

path_weights = 'model_final.pth'

# 변경이 필요한 부분
path_model_pretrained = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
model_title = path_model_pretrained.split("/")[1].split(".")[0]


# #### *Config 설정

# In[3]:


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(path_model_pretrained))
    cfg.OUTPUT_DIR = path_output
    cfg.DATASETS.TEST = (coco_dataset_test,)

    cfg.DATALOADER.NUM_WOREKRS = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6

    cfg.MODEL.RPN.NMS_THRESH = 0.5

    return cfg


# #### 데이터셋 등록

# In[4]:


if coco_dataset_test not in DatasetCatalog.list():
    register_coco_instances(coco_dataset_test, {}, path_dataset + 'test.json', path_dataset)

cfg = setup_cfg()


# In[5]:


def MyMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict

# test loader
test_loader = build_detection_test_loader(cfg, coco_dataset_test, MyMapper)


# In[6]:


# 각 fold 모델의 예측을 받아 평균을 냄
def ensemble_predictions(fold_outputs):
    final_scores = np.mean([fold['scores'] for fold in fold_outputs], axis=0)
    final_boxes = np.mean([fold['boxes'] for fold in fold_outputs], axis=0)
    final_targets = np.mean([fold['targets'] for fold in fold_outputs], axis=0)

    final_targets = np.round(final_targets).astype(int)
    
    return final_targets, final_boxes, final_scores


# In[7]:


prediction_strings = []
file_names = []

for data in tqdm(test_loader):
    
    prediction_string = ''
    fold_outputs = []
    data = data[0]
    
    for fold_idx in range(5):

        path_weight = os.path.join(f'{cfg.OUTPUT_DIR}_{fold_idx}', path_weights)
        
        if not os.path.exists(path_weight):    
            print("파일 없음", path_weight)
            continue
        
        cfg.MODEL.WEIGHTS = path_weight
        predictor = DefaultPredictor(cfg)

        outputs = predictor(data['image'])['instances']

        fold_outputs.append({
            'targets': outputs.pred_classes.cpu().tolist(),
            'boxes': [i.cpu().detach().numpy() for i in outputs.pred_boxes],
            'scores': outputs.scores.cpu().tolist(),
        })

    # 앙상블 예측값 계산
    targets, boxes, scores = ensemble_predictions(fold_outputs)
    
    for target, box, score in zip(targets, boxes, scores):
        prediction_string += f"{target} {score} {box[0]} {box[1]} {box[2]} {box[3]} "

    prediction_strings.append(prediction_string)
    file_names.append(data['file_name'])


# In[ ]:


submission = pd.DataFrame({
    'PredictionString': prediction_strings,
    'image_id': file_names
})
submission.to_csv(os.path.join('/data/ephemeral/home/baseline/custom', f'submission_{model_title}.csv'), index=False)

