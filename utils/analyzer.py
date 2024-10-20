import json
import numpy as np
from collections import defaultdict


def iou(gt_bbox, pred_bbox):
    x1_gt, y1_gt, w_gt, h_gt = gt_bbox
    x2_gt, y2_gt = x1_gt + w_gt, y1_gt + h_gt
    
    x1_pred, y1_pred, w_pred, h_pred = pred_bbox
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred
    
    x1_int = max(x1_gt, x1_pred)
    y1_int = max(y1_gt, y1_pred)
    x2_int = min(x2_gt, x2_pred)
    y2_int = min(y2_gt, y2_pred)
    
    intersection_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    
    gt_area = w_gt * h_gt
    pred_area = w_pred * h_pred
    
    union_area = gt_area + pred_area - intersection_area
    
    iou_value = intersection_area / union_area if union_area != 0 else 0
    return iou_value

# BBox 정답 여부 확인 함수
def is_bbox_correct(gt_bbox, pred_bbox, gt_category_id, pred_category_id, iou_threshold=0.5):
    iou_value = iou(gt_bbox, pred_bbox)
    if iou_value >= iou_threshold and gt_category_id == pred_category_id:
        return True
    return False


def load_gt_data(train_json_path):
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    gt_annotations = train_data['annotations']
    images_info = {image['id']: image['file_name'] for image in train_data['images']}
    return gt_annotations, images_info


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
        pred_bbox = pred['bbox']
        pred_category_id = pred['category_id']

        # 바운딩 박스의 면적 계산
        width, height = pred_bbox[2], pred_bbox[3]
        area = width * height

        
        gt_boxes = [ann for ann in gt_annotations if ann['image_id'] == image_id]

        for gt in gt_boxes:
            gt_bbox = gt['bbox']  
            gt_category_id = gt['category_id']


            if is_bbox_correct(gt_bbox, pred_bbox, gt_category_id, pred_category_id, iou_threshold):
                if area <= small_threshold:
                    correct_sml[gt_category_id]['S'] += 1
                elif small_threshold < area <= medium_threshold:
                    correct_sml[gt_category_id]['M'] += 1
                else:
                    correct_sml[gt_category_id]['L'] += 1

            if area <= small_threshold:
                total_sml[gt_category_id]['S'] += 1
            elif small_threshold < area <= medium_threshold:
                total_sml[gt_category_id]['M'] += 1
            else:
                total_sml[gt_category_id]['L'] += 1

    return correct_sml, total_sml


def calculate_actual_bbox_sml(gt_annotations, small_threshold, medium_threshold):
    actual_sml_counts = defaultdict(lambda: {'S': 0, 'M': 0, 'L': 0})

    for annotation in gt_annotations:
        bbox = annotation['bbox']
        class_id = annotation['category_id']
        width, height = bbox[2], bbox[3]
        area = width * height

        if area <= small_threshold:
            actual_sml_counts[class_id]['S'] += 1
        elif area <= medium_threshold:
            actual_sml_counts[class_id]['M'] += 1
        else:
            actual_sml_counts[class_id]['L'] += 1

    return actual_sml_counts


# 성능 평가 함수 (S, M, L별 맞은 바운딩 박스 계산)
def evaluate_bbox_performance_sml(train_json_path, val_bbox_json_path, small_threshold, medium_threshold, iou_threshold=0.5, output_file="bbox_performance.json"):
    print("계산중... 시간 좀 걸림...")
    gt_annotations, images_info = load_gt_data(train_json_path)
    predictions = load_pred_data(val_bbox_json_path)

    actual_sml_counts = calculate_actual_bbox_sml(gt_annotations, small_threshold, medium_threshold)    
    correct_sml, total_sml = compare_gt_pred_sml(gt_annotations, predictions, small_threshold, medium_threshold, iou_threshold)

    result_dict = {}
    for category_id in correct_sml.keys():
        result_dict[category_id] = {
            'S': {
                'matched': correct_sml[category_id]['S'],
                'total': total_sml[category_id]['S'],
                'gt': actual_sml_counts[category_id]['S']
            },
            'M': {
                'matched': correct_sml[category_id]['M'],
                'total': total_sml[category_id]['M'],
                'gt': actual_sml_counts[category_id]['M']
            },
            'L': {
                'matched': correct_sml[category_id]['L'],
                'total': total_sml[category_id]['L'],
                'gt': actual_sml_counts[category_id]['L']
            }
        }

    # 결과를 JSON 파일로 저장
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)

    print(f"Results saved to {output_file}")



root = '/data/ephemeral/home'
train_json_path = root + '/dataset/train.json'
val_bbox_json_path = root + '/outputs/val.pafpn.json'


small_threshold = 32 ** 2  
medium_threshold = 96 ** 2 

evaluate_bbox_performance_sml(train_json_path, val_bbox_json_path, small_threshold, medium_threshold, iou_threshold=0.5, output_file=root+"/outputs/bbox_performance.json")
