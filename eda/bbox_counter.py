import re
import os
import json
import pandas as pd



# json 파일에서 bbox 성능 데이터를 불러오는 함수
def load_bbox_performance(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# AP 값을 가져오는 함수
def get_ap_value(class_id, ap_values):
    return ap_values.get(class_id, {}).get('AP', 0.0)


# val_analysis 파일에서 AP 값을 추출하는 함수
def parse_val_analysis(file_path):
    ap_data = {}

    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = r"Class: (.+) \(ID: (\d+)\) - AP: ([\d.]+)"
    matches = re.findall(pattern, content)

    for match in matches:
        class_name = match[0]
        class_id = int(match[1])
        ap_value = float(match[2])
        ap_data[class_id] = {"class_name": class_name, "AP": ap_value}
    
    return ap_data



def fill_data_from_performance(bbox_performance, ap_values):
    data = {
        'Class': [0, 2, 7, 6, 9, 5, 1, 3, 4, 8],
        'S맞음': [],
        'S전체': [],
        'S실제': [],
        'M맞음': [],
        'M전체': [],
        'M실제': [],
        'L맞음': [],
        'L전체': [],
        'L실제': [],
        'AP': []
    }

    for class_id in data['Class']:
        class_data = bbox_performance.get(str(class_id), {})
        

        data['S맞음'].append(class_data.get('S', {}).get('matched', 0))
        data['S전체'].append(class_data.get('S', {}).get('total', 0))
        data['S실제'].append(class_data.get('S', {}).get('gt', 0))
        
        data['M맞음'].append(class_data.get('M', {}).get('matched', 0))
        data['M전체'].append(class_data.get('M', {}).get('total', 0))
        data['M실제'].append(class_data.get('M', {}).get('gt', 0))
        
        data['L맞음'].append(class_data.get('L', {}).get('matched', 0))
        data['L전체'].append(class_data.get('L', {}).get('total', 0))
        data['L실제'].append(class_data.get('L', {}).get('gt', 0))
        
        data['AP'].append(get_ap_value(class_id, ap_values))
    
    return data


def parse_log_stats(file_path):
    fail_sml = {}

    if not os.path.exists(file_path):
        print(f"{file_path} 파일을 찾을 수 없습니다.")
        return fail_sml

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # GT Category별 실패 개수 파싱
        for line in lines:
            line = line.strip()
            if line.startswith("Total missed bboxes"):
                break  
            if "," in line and line[0].isdigit():
                values = list(map(int, line.split(',')))
                gt_category_id = values[0]
                detection_failed = values[1:4]  # Detection Failed (S, M, L)
                classification_failed = values[4:]  # Classification Failed (S, M, L)

                fail_sml[gt_category_id] = detection_failed + classification_failed

    return fail_sml