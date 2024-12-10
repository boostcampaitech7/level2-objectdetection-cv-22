import json
import os

"""
    원본 데이터에서 특정 클래스의 데이터만 추출
    json 파일은 5개 카테고리("info", "licenses", "images", "annotations", "categories")
    
    불러 올 파일의 경로 : path_dataset

    불러 올 파일의 이름 : input_json_path
    저장할 파일의 이름 : output_json_path

    따로 추출할 데이터의 key(column) : column
    따로 추출할 데이터 키워드 : keyword
"""
# 경로 ─────────────────────────────────────────────────────────────
column = 'category_id'
keyword = 0

path_dataset = '/data/ephemeral/home/dataset/'

input_json_path = path_dataset + "train.json"
output_json_path = path_dataset + f'train_{column}_{keyword}.json'

# ──────────────────────────────────────────────────────────────────


with open(input_json_path, 'r') as f:
    data = json.load(f)


# column이 keyword인 annotations 필터링하여 image_id 추출

keyword_annotations = [ann for ann in data['annotations'] if ann[column] == keyword]

keyword_image_ids = set(ann['image_id'] for ann in keyword_annotations)

keyword_images = [img for img in data['images'] if img['id'] in keyword_image_ids]


# 새로운 JSON 데이터 구성

keyword_data = {
    'info': data.get('info', {}),
    'licenses': data.get('licenses', []),
    'images': keyword_images,
    'annotations': keyword_annotations,
    'categories': data.get('categories', [])
}


with open(output_json_path, 'w') as f:
    json.dump(keyword_data, f, indent=4)

print(f"필터링된 데이터가 {output_json_path}에 저장됨")
