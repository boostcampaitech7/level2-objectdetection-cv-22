import os
import json

path_dataset = '/data/ephemeral/home/dataset/'

def exclude_fold_from_train(input_train_json, input_fold_json, output_json):
    
    with open(input_train_json, 'r') as f:
        train_data = json.load(f)

    
    with open(input_fold_json, 'r') as f:
        fold_data = json.load(f)

    fold_image_ids = {image['id'] for image in fold_data['images']}
    
    remaining_images = [img for img in train_data['images'] if img['id'] not in fold_image_ids]
    remaining_annotations = [ann for ann in train_data['annotations'] if ann['image_id'] not in fold_image_ids]

    remaining_data = {
        "info": train_data.get("info", {}),
        "licenses": train_data.get("licenses", []),
        "images": remaining_images,
        "annotations": remaining_annotations,
        "categories": train_data.get("categories", [])
    }
    
    with open(output_json, 'w') as f:
        json.dump(remaining_data, f, indent=4)

    print(f"Exclusion complete. Result saved to {output_json}")

input_train_json = path_dataset + "train.json"
input_fold_json = path_dataset + "train_fold_3.json"

output_json = path_dataset + "train_merge.json"

exclude_fold_from_train(input_train_json, input_fold_json, output_json)
