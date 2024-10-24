import pickle
import pandas as pd

with open('/data/ephemeral/home/level2-objectdetection-cv-22/mmdetection/work_dirs/faster-rcnn_efb3_fpn_1x_coco.pkl', 'rb') as file:
    predictions = pickle.load(file)

prediction_strings = []
image_ids = []

for prediction in predictions:
    prediction_string = ''
    for bbox, score, label in zip(prediction['pred_instances']['bboxes'], 
                                  prediction['pred_instances']['scores'], 
                                  prediction['pred_instances']['labels']):
        prediction_string += f"{label.item()} {score.item()} {bbox[0].item()} {bbox[1].item()} {bbox[2].item()} {bbox[3].item()} "

    prediction_strings.append(prediction_string.strip())
    image_ids.append('test/' + prediction['img_path'].split('/')[-1]) 

submission_df = pd.DataFrame({
    'PredictionString': prediction_strings,
    'image_id': image_ids
})

submission_df.to_csv('/data/ephemeral/home/level2-objectdetection-cv-22/mmdetection/work_dirs/retinanet_x101-32x4d_fpn_1x_coco/eb3_pkl_to_csv.csv', index=False)