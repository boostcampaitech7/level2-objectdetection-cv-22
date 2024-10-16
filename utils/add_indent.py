import json

path_dataset = '/data/ephemeral/home/dataset/'
def format_json_with_indent(input_json, output_json, indent=4):
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"Formatted JSON saved to {output_json}")

# 사용 예시
input_json = path_dataset + 'train_fold_3.json'
output_json = path_dataset + 'train_fold_3_.json'

format_json_with_indent(input_json, output_json)