import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import os
from collections import defaultdict
import csv


def load_json_results(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def load_annotations(train_json_path):
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    return data['annotations'], data['images']


def load_map_values(csv_file_path):
    map_values = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_id = int(row['image_id'])
            map_value = float(row['mAP'])
            map_values[image_id] = map_value
    return map_values


def iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_min = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def visualize_detection(results, annotations, images_info, dataset_dir, output_dir, map_file_path, log_file_path):
    
    category_names = {0: 'General trash', 1: 'Paper', 2: 'Paper pack', 3: 'Metal', 
                  4: 'Glass', 5: 'Plastic', 6: 'Styrofoam', 7: 'Plastic bag', 
                  8: 'Battery', 9: 'Clothing'}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    map_values = load_map_values(map_file_path)

    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result["image_id"]].append(result)
    
    grouped_annotations = defaultdict(list)
    for annotation in annotations:
        grouped_annotations[annotation["image_id"]].append(annotation)

    image_id_to_filename = {image['id']: image['file_name'] for image in images_info}

    
    with open(log_file_path, 'w') as log_file:
        log_file.write("Missed Bboxes (Image ID, Bbox ID, Category ID):\n")

        for image_id, result_list in grouped_results.items():
            image_file_name = image_id_to_filename.get(image_id, None)

            if image_file_name is None:
                print(f"{image_id} 이미지 없음")
                continue

            image_file_name = f"{str(image_id).zfill(4)}.jpg"
            image_path = os.path.join(dataset_dir, image_file_name)

            if not os.path.exists(image_path):
                print(f"{image_file_name} 파일 없음")
                continue

            image = Image.open(image_path)

            fig, ax = plt.subplots(1)
            ax.imshow(image)

            # Ground Truth(파랑)
            ground_truth_boxes = []
            if image_id in grouped_annotations:
                map_value = map_values.get(image_id, None)

                for annotation in grouped_annotations[image_id]:
                    bbox = annotation["bbox"]
                    category_id = annotation["category_id"]
                    
                    x_min, y_min, width, height = bbox
                    ground_truth_boxes.append((bbox, category_id))

                    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='b', facecolor='none', linestyle='--', label="Ground Truth")
                    ax.add_patch(rect)

                    label = f"{category_names[category_id]}"
                    plt.text(x_min + width, y_min, label, color='white', backgroundcolor='blue', fontsize=6, alpha=0.8)

            # 예측 바운딩 박스(빨강)
            for result in result_list:
                pred_bbox = result["bbox"]
                score = result["score"]
                pred_category_id = result["category_id"]
                
                x_min, y_min, width, height = pred_bbox
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none', label="Prediction")
                ax.add_patch(rect)

                label = f"{category_names[pred_category_id]}: {score:.2f}"
                plt.text(x_min, y_min - 5, label, color='white', backgroundcolor='red', fontsize=6, alpha=0.8)

                # 탐지 및 분류 실패 판단
                detection_failure = True
                classification_failure = False

                for gt_bbox, gt_category_id in ground_truth_boxes:
                    iou_value = iou(pred_bbox, gt_bbox)

                    if iou_value >= 0.5:
                        detection_failure = False
                        if pred_category_id != gt_category_id:
                            classification_failure = True
                        break

                # 실패 기록 및 이미지에 텍스트 추가
                if detection_failure:
                    plt.text(x_min, y_min + height, 'detection failed', color='white', backgroundcolor='pink', fontsize=6, alpha=0.8)
                    log_file.write(f"{image_id}, {result['id']}, {pred_category_id} (Detection Failed)\n")
                    
                elif classification_failure:
                    plt.text(x_min + 10, y_min + height, 'classification failed', color='white', backgroundcolor='orange', fontsize=6, alpha=0.8)
                    log_file.write(f"{image_id}, {result['id']}, {pred_category_id} (Classification Failed)\n")

                if map_value is not None:
                    map_label = f"mAP: {map_value:.2f}"
                    plt.text(10, 10, map_label, color='white', backgroundcolor='green', fontsize=10)


            if map_value is not None and map_value < 0.5:
                output_path = os.path.join(output_dir, f"{image_id}_detection.jpg")
                plt.axis("off")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                count += 1
                print(f"{output_path}에 저장됨")


        log_file.write(f"\nTotal missed bboxes: {count}\n")

    return count

