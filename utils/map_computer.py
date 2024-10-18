from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv


train_json_path = '/data/ephemeral/home/dataset/train.json'
val_json_path = '/data/ephemeral/home/outputs/val.bbox.json'

cocoGt = COCO(train_json_path)
cocoDt = cocoGt.loadRes(val_json_path)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

image_map_values = {}

val_image_ids = cocoDt.getImgIds()
train_image_ids = cocoGt.getImgIds()

for image_id in val_image_ids:
    cocoEval.params.imgIds = [image_id]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    map_value = cocoEval.stats[0]
    
    if map_value > 0:
        image_map_values[image_id] = map_value
        

output_csv_path = '/data/ephemeral/home/outputs/output_map.csv'
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "mAP"])
    
    for image_id, map_value in image_map_values.items():
        writer.writerow([image_id, map_value])

print(f"{output_csv_path}저장 완료")
