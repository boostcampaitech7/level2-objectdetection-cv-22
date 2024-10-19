import json

"""
    띄어쓰기(indent=4) 반영하여 json 파일을 다시 저장
    
    불러 올 파일의 경로 : path_dataset
    불러 올 파일의 이름 : input_json
    저장할 파일의 이름 : output_json
"""
# 경로 ─────────────────────────────────────────────────────────────

path_dataset = '/data/ephemeral/home/dataset/'

input_json = path_dataset + 'val_fold_3.json'
output_json = path_dataset + 'val_fold_3_.json'

# ──────────────────────────────────────────────────────────────────


def format_json_with_indent(input_json, output_json, indent=4):
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"Formatted JSON saved to {output_json}")


format_json_with_indent(input_json, output_json)