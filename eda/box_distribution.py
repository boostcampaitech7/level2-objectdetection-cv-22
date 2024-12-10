import matplotlib.pyplot as plt
import numpy as np
from k_means import load_boxes_from_coco

# 바운딩 박스 분포 시각화
def plot_box_distribution(boxes):
    widths = boxes[:, 0]
    heights = boxes[:, 1]
    ratios = widths / heights
    
    # 크기 분포 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(widths, heights, alpha=0.3)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Bounding Box Size Distribution')

    # 비율 분포 시각화
    plt.subplot(1, 2, 2)
    plt.hist(ratios, bins=30, color='green', alpha=0.7)
    plt.xlabel('Aspect Ratio (Width/Height)')
    plt.ylabel('Frequency')
    plt.title('Bounding Box Aspect Ratio Distribution')

    plt.tight_layout()
    plt.show()

# 바운딩 박스 데이터 로드 후 시각화
if __name__ == "__main__":
    # COCO 포맷에서 바운딩 박스 데이터 로드
    root = 'C:/Users/yeyec/workspace/server2'
    boxes = load_boxes_from_coco(root + '/dataset/train.json')
    
    # 바운딩 박스 분포 시각화
    plot_box_distribution(boxes)
