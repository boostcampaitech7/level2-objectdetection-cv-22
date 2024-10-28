import json
import numpy as np


def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


# K-means를 사용한 Anchor Box 결정 함수
def kmeans(boxes, k, dist=np.median, seed=42):
    np.random.seed(seed)
    clusters = boxes[np.random.choice(boxes.shape[0], k, replace=False)]
    
    while True:
        distances = np.array([1 - iou(box, clusters) for box in boxes])
        nearest_clusters = np.argmin(distances, axis=1)
        new_clusters = np.array([dist(boxes[nearest_clusters == i], axis=0) for i in range(k)])
        
        if (clusters == new_clusters).all():
            break
        
        clusters = new_clusters
    
    return clusters


def load_boxes_from_coco(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    boxes = []
    for annotation in data['annotations']:
        _, _, width, height = annotation['bbox']
        boxes.append([width, height])
    
    return np.array(boxes)


# 앵커 박스 크기 정수화 및 비율 계산 함수
def calculate_anchor_ratios(clusters):
    clusters = np.round(clusters).astype(int)
    ratios = clusters[:, 0] / clusters[:, 1]
    
    return clusters, ratios


if __name__ == "__main__":
    root = 'C:/Users/yeyec/workspace/server2'
    boxes = load_boxes_from_coco(root + '/dataset/train.json')
    
    # K 값 (Anchor Box의 수) 설정
    k = 5
    
    # K-means 적용
    clusters = kmeans(boxes, k)
    
    # 앵커 박스 정수화 및 비율 계산
    clusters, ratios = calculate_anchor_ratios(clusters)
    
    # 결과 출력
    print("Anchor Boxes (Width, Height) and Ratios:")
    for i, (width, height, ratio) in enumerate(zip(clusters[:, 0], clusters[:, 1], ratios)):
        print(f"Anchor Box {i+1}: Width={width}, Height={height}, Ratio={ratio:.2f}")
