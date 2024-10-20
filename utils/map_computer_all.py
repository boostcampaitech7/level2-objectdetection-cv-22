import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def analyze_val_results(gt_file, pred_file, output_file):
    
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    with open(output_file, 'w') as f:
        f.write("COCO Evaluation Results\n")
        f.write("=======================\n")
        
        f.write(f"AP (Average Precision) @[IoU=0.50:0.95]: {coco_eval.stats[0]:.3f}\n")
        f.write(f"AP50 (IoU=0.50): {coco_eval.stats[1]:.3f}\n")
        f.write(f"AP75 (IoU=0.75): {coco_eval.stats[2]:.3f}\n")
        f.write(f"AP (small objects): {coco_eval.stats[3]:.3f}\n")
        f.write(f"AP (medium objects): {coco_eval.stats[4]:.3f}\n")
        f.write(f"AP (large objects): {coco_eval.stats[5]:.3f}\n")
        f.write(f"AR (Average Recall) @[IoU=0.50:0.95]: {coco_eval.stats[6]:.3f}\n")
        f.write(f"AR (small objects): {coco_eval.stats[7]:.3f}\n")
        f.write(f"AR (medium objects): {coco_eval.stats[8]:.3f}\n")
        f.write(f"AR (large objects): {coco_eval.stats[9]:.3f}\n\n")

        f.write("Per-class Evaluation\n")
        f.write("====================\n")

        for cat_id in coco_gt.getCatIds():
            class_name = coco_gt.loadCats(cat_id)[0]['name']
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            ap = coco_eval.stats[0]
            f.write(f"Class: {class_name} (ID: {cat_id}) - AP: {ap:.3f}\n")


    print(f"Results saved in {output_file}")

root = '/data/ephemeral/home'
gt_file = root + '/dataset/val_fold_3.json'
pred_file = root + '/outputs/val.bbox.json'
output_file = root + '/outputs/val_analysis.txt'

analyze_val_results(gt_file, pred_file, output_file)

