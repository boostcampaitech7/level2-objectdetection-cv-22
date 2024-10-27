import json
import numpy as np
from k_means import load_boxes_from_coco, iou

def mean_iou(boxes, anchors):
    """
    Calculate the mean IoU between dataset bounding boxes and anchor boxes.
    :param boxes: numpy array, shape=(N, 2), bounding boxes (width, height)
    :param anchors: numpy array, shape=(K, 2), anchor boxes (width, height)
    :return: float, mean IoU value
    """
    ious = []
    for box in boxes:
        # 각 박스와 모든 앵커박스 간의 IoU를 계산
        iou_values = [np.max(iou(box, anchors))]
        ious.append(np.mean(iou_values))
    
    return np.mean(ious)

# 앵커박스 SIZES와 비율을 사용해 (Width, Height) Anchor Box 생성
def generate_anchors_from_sizes_and_ratios(sizes, aspect_ratios):
    """
    Generate (Width, Height) anchor boxes using sizes and aspect ratios.
    :param sizes: list of sizes (scales)
    :param aspect_ratios: list of aspect ratios
    :return: numpy array of (Width, Height) anchor boxes
    """
    anchors = []
    for size in sizes:
        for ratio in aspect_ratios:
            # Aspect Ratio를 사용해 너비와 높이 계산
            width = int(size * np.sqrt(ratio))
            height = int(size / np.sqrt(ratio))
            anchors.append([width, height])
    return np.array(anchors)

# 데이터 로드 및 Mean IoU 계산 예제
if __name__ == "__main__":
    # COCO 포맷에서 바운딩 박스 데이터 로드
    root = 'C:/Users/yeyec/workspace/server2'
    boxes = load_boxes_from_coco(root + '/dataset/train.json')
    
    # baseline : Mean IoU: 0.67
    sizes = [32, 64, 128, 256, 512]
    aspect_ratios = [0.5, 1.0, 1.5]

    # k-means Mean IoU: 0.64
    #sizes = [314, 102, 50, 541, 184]
    #aspect_ratios = [0.9, 1.1, 1.2]

    # k-means 보정1 Mean IoU: 0.68
    #sizes = [314, 102, 50, 541, 184]
    #aspect_ratios = [0.7, 1.1, 1.4]

    # k-means 보정2 Mean IoU: 0.70
    #sizes = [314, 102, 45, 541, 184]
    #aspect_ratios = [0.6, 0.8, 1.2, 1.8]

    # Mean IoU: 0.71
    #sizes = [28, 45, 90, 165, 314, 541]
    #aspect_ratios = [0.5, 1.0, 1.8, 4.0]

    # sml에 따른 크기 Mean IoU: 0.69
    #sizes = [32, 64, 128, 256, 512]
    #aspect_ratios = [0.5, 1.0, 1.5, 2.0, 4.0]
    
    
    # (Width, Height) 형태의 Anchor Box 생성
    anchors = generate_anchors_from_sizes_and_ratios(sizes, aspect_ratios)
    
    # Mean IoU 계산
    mean_iou_value = mean_iou(boxes, anchors)
    print(f"Mean IoU: {mean_iou_value:.2f}")
