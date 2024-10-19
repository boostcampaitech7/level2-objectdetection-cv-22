import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def draw_bbox(image, bbox, label, color='blue', line_width=2):
    x_min, y_min, width, height = bbox
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=line_width, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x_min, y_min - 5, label, color='white', backgroundcolor=color, fontsize=8, alpha=0.8)


def calculate_iou(gt_bbox, pred_bbox):
    x1_gt, y1_gt, w_gt, h_gt = gt_bbox
    x1_pred, y1_pred, w_pred, h_pred = pred_bbox
    
    x2_gt, y2_gt = x1_gt + w_gt, y1_gt + h_gt
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred
    
    # 교차 영역 좌표
    xi1, yi1 = max(x1_gt, x1_pred), max(y1_gt, y1_pred)
    xi2, yi2 = min(x2_gt, x2_pred), min(y2_gt, y2_pred)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height
    
    # 각각의 영역 계산
    gt_area = w_gt * h_gt
    pred_area = w_pred * h_pred
    
    union = gt_area + pred_area - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou


def visualize_single_image(images_dir, annotations, output_dir, categories_id_name, selected_image_id=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

    if selected_image_id is None:
        print("Please provide an image_id.")
        return

    image_filename = f"{str(selected_image_id).zfill(4)}.jpg"

    image_path = os.path.join(images_dir, image_filename)
    print(f"Full image path: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found in {images_dir}.")
        return

    image = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')

    annotations_for_image = [ann for ann in annotations['annotations'] if ann['image_id'] == selected_image_id]

    for annotation in annotations_for_image:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        label = categories_id_name.get(category_id, f'Class {category_id}')
        draw_bbox(image, bbox, label, color='blue')

    output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_gt.jpg")
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {output_image_path}")


# 탐지 실패된 바운딩 박스 시각화 및 저장 함수
def visualize_detection_failed(images_dir, annotations, predictions, output_dir, categories_id_name, iou_threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

    for pred in predictions:
        image_id = pred['image_id']
        pred_bbox = pred['bbox']
        pred_category_id = pred['category_id']
        pred_score = pred['score']


        gt_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        detection_failed = True

        for gt_ann in gt_annotations:
            gt_bbox = gt_ann['bbox']
            gt_category_id = gt_ann['category_id']

            iou = calculate_iou(gt_bbox, pred_bbox)

            # IoU가 기준 이상이고, 클래스가 같으면 탐지 성공
            if iou >= iou_threshold and pred_category_id == gt_category_id:
                detection_failed = False
                break

        
        if detection_failed:
            image_filename = image_id_to_filename.get(image_id, None)
            if image_filename is None:
                continue

            image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Image {image_filename} not found.")
                continue

            image = Image.open(image_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.axis('off')

            
            label = f"{categories_id_name.get(pred_category_id, 'Unknown')}: {pred_score:.2f}"
            draw_bbox(image, pred_bbox, label, color='red')

            
            output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_detection_failed.jpg")
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved: {output_image_path}")


def visualize_detection_failed_single_image(images_dir, annotations, predictions, output_dir, categories_id_name, selected_image_id, iou_threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

    image_filename = image_id_to_filename.get(selected_image_id, None)
    if image_filename is None:
        print(f"Image ID {selected_image_id} not found.")
        return

    image_filename = f"{str(selected_image_id).zfill(4)}.jpg"
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found in {images_dir}.")
        return

    pred_list = [pred for pred in predictions if pred['image_id'] == selected_image_id]
    gt_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == selected_image_id]

    image = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')

    detection_failed = False

    for pred in pred_list:
        pred_bbox = pred['bbox']
        pred_category_id = pred['category_id']
        pred_score = pred['score']

        # 탐지 실패 여부를 확인
        for gt_ann in gt_annotations:
            gt_bbox = gt_ann['bbox']
            gt_category_id = gt_ann['category_id']

            iou = calculate_iou(gt_bbox, pred_bbox)
            if iou >= iou_threshold and pred_category_id == gt_category_id:
                break
        else:
            detection_failed = True
            label = f"{categories_id_name.get(pred_category_id, 'Unknown')}: {pred_score:.2f}"
            draw_bbox(image, pred_bbox, label, color='red')

    if detection_failed:
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_detection_failed.jpg")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {output_image_path}")
    else:
        print(f"No detection failed for image ID {selected_image_id}")


def visualize_correct_bboxes_single_image(images_dir, annotations, predictions, output_dir, categories_id_name, selected_image_id, iou_threshold=0.5):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_filename = f"{str(selected_image_id).zfill(4)}.jpg"
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found in {images_dir}.")
        return

    pred_list = [pred for pred in predictions if pred['image_id'] == selected_image_id]

    gt_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == selected_image_id]

    image = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')

    correct_detected = False
    correct_bbox_count = 0

    for pred in pred_list:
        pred_bbox = pred['bbox']
        pred_category_id = pred['category_id']
        pred_score = pred['score']

        # 정답(탐지 성공) 여부 확인
        for gt_ann in gt_annotations:
            gt_bbox = gt_ann['bbox']
            gt_category_id = gt_ann['category_id']

            iou = calculate_iou(gt_bbox, pred_bbox)
            if iou >= iou_threshold and pred_category_id == gt_category_id:
                correct_detected = True
                correct_bbox_count += 1
                label = f"{categories_id_name.get(pred_category_id, 'Unknown')}: {pred_score:.2f}"
                draw_bbox(image, pred_bbox, label, color='green')
                break

    if correct_detected:
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_correct_bboxes.jpg")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {output_image_path}")
    else:
        print(f"No correct detections for image ID {selected_image_id}")
    
    gt_bbox_count = len(gt_annotations)
    print(f"ID {selected_image_id} : GT {gt_bbox_count}개 / 정답 처리수 {correct_bbox_count}")



def find_image_filename(image_id, annotations):
    """Find image filename using image_id from annotations."""
    return f"{str(image_id).zfill(4)}.jpg"


def main():
    root = '/data/ephemeral/home'

    images_dir = root + '/dataset/train/'  
    annotation_file = root + '/dataset/val_fold_3.json'
    prediction_file = root + '/outputs/val.bbox.json'

    output_dir = root + '/outputs/ground_truth'
    output_det_dir = root + '/outputs/detection_failed'
    output_cor_dir = root + '/outputs/correct_bboxes'

    annotations = load_annotations(annotation_file)
    predictions = load_annotations(prediction_file)

    categories_id_name = {cat['id']: cat['name'] for cat in annotations['categories']}

    # 선택할 image_id 설정
    selected_image_id = 675  

    found_image_filename = find_image_filename(selected_image_id, annotations)

    if found_image_filename:
        print(f"Image filename for image_id {selected_image_id}: {found_image_filename}")
    else:
        print(f"No image found for image_id {selected_image_id}.")

    visualize_single_image(images_dir, annotations, output_dir, categories_id_name, selected_image_id)
    #visualize_detection_failed(images_dir, annotations, predictions, output_det_dir, categories_id_name)
    visualize_detection_failed_single_image(images_dir, annotations, predictions, output_det_dir, categories_id_name, selected_image_id)
    visualize_correct_bboxes_single_image(images_dir, annotations, predictions, output_cor_dir, categories_id_name, selected_image_id)


if __name__ == "__main__":
    main()
