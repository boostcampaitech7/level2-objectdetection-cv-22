import json
import numpy as np

# IoU 계산 함수
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
    """
    Run K-means algorithm to find k anchor boxes.
    :param boxes: numpy array, shape=(N, 2), N boxes with width and height
    :param k: int, number of clusters
    :param dist: function, function to calculate the distance (default: median)
    :param seed: int, random seed
    :return: numpy array, shape=(k, 2), width and height of k clusters
    """
    np.random.seed(seed)
    
    # 초기 중심(centroids) 랜덤 선택
    clusters = boxes[np.random.choice(boxes.shape[0], k, replace=False)]
    
    while True:
        # 각 박스에 대해 가장 가까운 클러스터 할당
        distances = np.array([1 - iou(box, clusters) for box in boxes])
        nearest_clusters = np.argmin(distances, axis=1)
        
        # 새로운 클러스터 계산
        new_clusters = np.array([dist(boxes[nearest_clusters == i], axis=0) for i in range(k)])
        
        # 변화가 없다면 종료
        if (clusters == new_clusters).all():
            break
        
        clusters = new_clusters
    
    return clusters

# COCO 형식의 JSON에서 바운딩 박스 정보 추출
def load_boxes_from_coco(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    boxes = []
    for annotation in data['annotations']:
        # COCO 포맷에서는 bbox가 [x, y, width, height]로 되어 있음
        _, _, width, height = annotation['bbox']
        boxes.append([width, height])
    
    return np.array(boxes)

# 앵커 박스 크기 정수화 및 비율 계산 함수
# 비율(Aspect Ratio) 계산 및 클러스터링
def kmeans_for_ratios(boxes, k=3, dist=np.median, seed=42):
    """
    Run K-means algorithm to find k representative aspect ratios.
    :param boxes: numpy array, shape=(N, 2), bounding boxes (width, height)
    :param k: int, number of clusters
    :param dist: function, function to calculate the distance (default: median)
    :param seed: int, random seed
    :return: numpy array, shape=(k,), representative aspect ratios
    """
    ratios = boxes[:, 0] / boxes[:, 1]  # 너비/높이 비율 계산
    ratios = ratios.reshape(-1, 1)  # K-means를 위해 2차원 배열로 변환
    
    # K-means 수행
    np.random.seed(seed)
    clusters = ratios[np.random.choice(ratios.shape[0], k, replace=False)]
    
    while True:
        distances = np.abs(ratios - clusters.T)  # 절대값 거리 사용
        nearest_clusters = np.argmin(distances, axis=1)
        
        new_clusters = np.array([dist(ratios[nearest_clusters == i]) for i in range(k)])
        
        if (clusters == new_clusters).all():
            break
        
        clusters = new_clusters
    
    return clusters.flatten()


def split_boxes_by_size(boxes, small_threshold=32*32, medium_threshold=96*96):
    """
    Split boxes into small, medium, and large based on area.
    :param boxes: numpy array, shape=(N, 2), bounding boxes (width, height)
    :param small_threshold: area threshold for small boxes
    :param medium_threshold: area threshold for medium boxes
    :return: small_boxes, medium_boxes, large_boxes
    """
    small_boxes = []
    medium_boxes = []
    large_boxes = []
    
    for box in boxes:
        area = box[0] * box[1]  # 너비 * 높이로 면적 계산
        
        if area <= small_threshold:
            small_boxes.append(box)
        elif area <= medium_threshold:
            medium_boxes.append(box)
        else:
            large_boxes.append(box)
    
    return np.array(small_boxes), np.array(medium_boxes), np.array(large_boxes)



if __name__ == "__main__":
    # COCO 포맷에서 바운딩 박스 데이터 로드
    root = 'C:/Users/yeyec/workspace/server2'
    boxes = load_boxes_from_coco(root + '/dataset/train.json')
    
    # 바운딩 박스를 S, M, L로 나누기
    small_boxes, medium_boxes, large_boxes = split_boxes_by_size(boxes)
    
    # K 값 설정 (각 그룹별로 개별 설정 가능)
    k_s, k_m, k_l = 3, 3, 3  # Small, Medium, Large에 대해 각각 3개의 클러스터 사용
    
    # 각 그룹에 대해 K-means 적용 (너비와 높이)
    small_anchors = kmeans(small_boxes, k_s) if len(small_boxes) > 0 else []
    medium_anchors = kmeans(medium_boxes, k_m) if len(medium_boxes) > 0 else []
    large_anchors = kmeans(large_boxes, k_l) if len(large_boxes) > 0 else []
    
    # 각 그룹에 대해 K-means 적용 (비율)
    small_ratios = kmeans_for_ratios(small_boxes, k=3) if len(small_boxes) > 0 else []
    medium_ratios = kmeans_for_ratios(medium_boxes, k=3) if len(medium_boxes) > 0 else []
    large_ratios = kmeans_for_ratios(large_boxes, k=3) if len(large_boxes) > 0 else []
    
    # 결과 출력
    print("Small Anchor Boxes (Width, Height) and Ratios:")
    for anchor in small_anchors:
        print(f"Width={int(anchor[0])}, Height={int(anchor[1])}")
    print(f"Small Ratios: {small_ratios}")
    
    print("\nMedium Anchor Boxes (Width, Height) and Ratios:")
    for anchor in medium_anchors:
        print(f"Width={int(anchor[0])}, Height={int(anchor[1])}")
    print(f"Medium Ratios: {medium_ratios}")
    
    print("\nLarge Anchor Boxes (Width, Height) and Ratios:")
    for anchor in large_anchors:
        print(f"Width={int(anchor[0])}, Height={int(anchor[1])}")
    print(f"Large Ratios: {large_ratios}")
