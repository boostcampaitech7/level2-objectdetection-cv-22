import json
import numpy as np
from collections import defaultdict

# IoU 계산 함수
def iou(gt_bbox, pred_bbox):
    x1_gt, y1_gt, w_gt, h_gt = gt_bbox
    x2_gt, y2_gt = x1_gt + w_gt, y1_gt + h_gt
    
    x1_pred, y1_pred, w_pred, h_pred = pred_bbox
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred
    
    # 교집합 영역 좌표 계산
    x1_int = max(x1_gt, x1_pred)
    y1_int = max(y1_gt, y1_pred)
    x2_int = min(x2_gt, x2_pred)
    y2_int = min(y2_gt, y2_pred)
    
    # 교집합 넓이 계산
    intersection_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    
    # 각 박스의 넓이 계산
    gt_area = w_gt * h_gt
    pred_area = w_pred * h_pred
    
    # 합집합 넓이 계산
    union_area = gt_area + pred_area - intersection_area
    
    # IoU 계산
    iou_value = intersection_area / union_area if union_area != 0 else 0
    return iou_value

# BBox 정답 여부 확인 함수
def is_bbox_correct(gt_bbox, pred_bbox, gt_category_id, pred_category_id, iou_threshold=0.5):
    iou_value = iou(gt_bbox, pred_bbox)
    if iou_value >= iou_threshold and gt_category_id == pred_category_id:
        return True  # 정답
    return False  # 오답

# train.json에서 Ground Truth 데이터 로드
def load_gt_data(train_json_path):
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    gt_annotations = train_data['annotations']  # GT 바운딩 박스 정보
    images_info = {image['id']: image['file_name'] for image in train_data['images']}  # 이미지 ID와 파일 이름 매핑
    return gt_annotations, images_info

# val.bbox.json에서 예측 데이터 로드
def load_pred_data(val_bbox_json_path):
    with open(val_bbox_json_path, 'r') as f:
        val_data = json.load(f)

    if isinstance(val_data, list):
        predictions = val_data
    else:
        predictions = val_data.get('predictions', [])

    return predictions

# GT와 예측 BBox 비교 및 성능 평가 (S, M, L별 맞은 바운딩 박스 계산)
def compare_gt_pred_sml(gt_annotations, predictions, small_threshold, medium_threshold, iou_threshold=0.5):
    correct_sml = defaultdict(lambda: {'S': 0, 'M': 0, 'L': 0})
    total_sml = defaultdict(lambda: {'S': 0, 'M': 0, 'L': 0})

    for pred in predictions:
        image_id = pred['image_id']
        pred_bbox = pred['bbox']  # 예측된 바운딩 박스
        pred_category_id = pred['category_id']  # 예측된 클래스 ID

        # 바운딩 박스의 면적 계산
        width, height = pred_bbox[2], pred_bbox[3]
        area = width * height

        # 해당 이미지의 GT (Ground Truth) 바운딩 박스 가져오기
        gt_boxes = [ann for ann in gt_annotations if ann['image_id'] == image_id]

        for gt in gt_boxes:
            gt_bbox = gt['bbox']  # GT 바운딩 박스
            gt_category_id = gt['category_id']  # GT 클래스 ID

            # 예측과 GT 바운딩 박스 비교
            if is_bbox_correct(gt_bbox, pred_bbox, gt_category_id, pred_category_id, iou_threshold):
                if area <= small_threshold:
                    correct_sml[gt_category_id]['S'] += 1
                elif small_threshold < area <= medium_threshold:
                    correct_sml[gt_category_id]['M'] += 1
                else:
                    correct_sml[gt_category_id]['L'] += 1

            # 전체 카운트 증가
            if area <= small_threshold:
                total_sml[gt_category_id]['S'] += 1
            elif small_threshold < area <= medium_threshold:
                total_sml[gt_category_id]['M'] += 1
            else:
                total_sml[gt_category_id]['L'] += 1

    return correct_sml, total_sml

# 성능 평가 함수 (S, M, L별 맞은 바운딩 박스 계산)
def evaluate_bbox_performance_sml(train_json_path, val_bbox_json_path, small_threshold, medium_threshold, iou_threshold=0.5):
    gt_annotations, images_info = load_gt_data(train_json_path)
    predictions = load_pred_data(val_bbox_json_path)

    # GT와 예측 바운딩 박스 비교 (S, M, L 별로)
    correct_sml, total_sml = compare_gt_pred_sml(gt_annotations, predictions, small_threshold, medium_threshold, iou_threshold)

    for category_id in correct_sml.keys():
        print(f"\nClass {category_id}:")
        print(f"  S: 맞은 개수: {correct_sml[category_id]['S']} / 전체: {total_sml[category_id]['S']}")
        print(f"  M: 맞은 개수: {correct_sml[category_id]['M']} / 전체: {total_sml[category_id]['M']}")
        print(f"  L: 맞은 개수: {correct_sml[category_id]['L']} / 전체: {total_sml[category_id]['L']}")


root = '/data/ephemeral/home'
train_json_path = root + '/dataset/train.json'
val_bbox_json_path = root + '/outputs/val.bbox.json'


small_threshold = 32 ** 2  # S 크기의 임계값
medium_threshold = 96 ** 2  # M 크기의 임계값

evaluate_bbox_performance_sml(train_json_path, val_bbox_json_path, small_threshold, medium_threshold, iou_threshold=0.5)
