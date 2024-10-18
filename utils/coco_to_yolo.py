import json
import os

def coco_to_yolo(annotation, image_width, image_height):
    """
    COCO 형식의 바운딩 박스를 YOLO 포맷으로 변환.
    :param annotation: COCO 형식의 어노테이션 (dict)
    :param image_width: 이미지의 너비
    :param image_height: 이미지의 높이
    :return: YOLO 형식의 어노테이션 (str)
    """
    category_id = annotation["category_id"]
    bbox = annotation["bbox"]  # [x, y, width, height]
    x, y, width, height = bbox

    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height

    return f"{category_id} {x_center} {y_center} {width_normalized} {height_normalized}\n"


def convert_coco_json_to_yolo(json_path, output_dir, images_info):
    """
    COCO JSON 파일에서 어노테이션을 가져와 YOLO 포맷으로 변환.
    :param json_path: COCO 형식의 JSON 파일 경로.
    :param output_dir: YOLO 어노테이션 파일을 저장할 디렉토리 경로.
    :param images_info: 이미지 정보를 담은 딕셔너리 (image_id -> (width, height)).
    """

    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']

    os.makedirs(output_dir, exist_ok=True)

    for annotation in annotations:
        image_id = annotation['image_id']
        image_info = images_info.get(image_id)

        if not image_info:
            print(f"Image ID {image_id} 정보 없음")
            continue

        image_width, image_height = image_info
        yolo_annotation = coco_to_yolo(annotation, image_width, image_height)

        output_file_path = os.path.join(output_dir, f"{image_id}.txt")

        with open(output_file_path, 'a') as f:
            f.write(yolo_annotation)


def get_images_info(data):
    images_info = {}
    for image in data['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        images_info[image_id] = (image_width, image_height)
    return images_info


json_path = '/data/ephemeral/home/dataset/train.json'  
output_dir = '/data/ephemeral/home/dataset/yolo'

with open(json_path, 'r') as f:
    data = json.load(f)


images_info = get_images_info(data)
convert_coco_json_to_yolo(json_path, output_dir, images_info)
