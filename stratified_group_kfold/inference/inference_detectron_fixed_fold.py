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

# 개인화
from stratified_group_kfold.inference.inference_utils import MyMapper

"""
    고정된 1개의 파일만 inference
    (전체 파일에 대한 시각화 포함)
"""

# ───────────────────────────────────────────────────────────────────────────────────────────

def setup_cfg(con22, filename):
    path_weight = os.path.join(f'{con22.path_output}{filename}', con22.filename_weights)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(con22.path_model_pretrained))
    cfg.OUTPUT_DIR = con22.path_output + filename
    cfg.DATASETS.TEST = (con22.coco_dataset_test,)
    
    cfg.MODEL.WEIGHTS = path_weight

    cfg.DATALOADER.NUM_WORKERS = con22.NUM_WORKERS
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = con22.ROI_HEADS_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = con22.ROI_HEADS_NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = con22.ROI_HEADS_SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = con22.ROI_HEADS_NMS_THRESH_TEST

    cfg.MODEL.RPN.NMS_THRESH = con22.RPN_NMS_THRESH

    return cfg


def test_model(con22, visualized=False, filename=''):

    prediction_strings = []
    file_names = []

    # 테스트 데이터셋 등록
    if con22.coco_dataset_test not in DatasetCatalog.list():
        register_coco_instances(
            con22.coco_dataset_test, {}, 
            os.path.join(con22.path_dataset, 'test.json'), 
            con22.path_dataset
        )

    cfg = setup_cfg(con22, filename)

    # 데이터 로더 및 예측기 설정
    test_loader = build_detection_test_loader(cfg, con22.coco_dataset_test, MyMapper, num_workers=4)
    predictor = DefaultPredictor(cfg)

    for data in tqdm(test_loader):
        prediction_string = ''
        data = data[0]

        # 예측 수행
        outputs = predictor(data['image'])['instances']

        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        # 결과를 문자열로 변환하여 저장
        for target, box, score in zip(targets, boxes, scores):
            prediction_string += f"{target} {score} {box[0]} {box[1]} {box[2]} {box[3]} "

        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace(con22.path_dataset, ''))

        # 시각화 옵션이 켜져 있는 경우 결과 이미지 저장
        if visualized:
            v = Visualizer(data['image'], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs.to("cpu"))

            result_file = os.path.join(cfg.OUTPUT_DIR, f"visualization_{data['file_name'].split('/')[-1]}")
            cv2.imwrite(result_file, out.get_image()[:, :, ::-1])
    
    # 제출 파일 생성
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det_{filename}.csv'), index=False)