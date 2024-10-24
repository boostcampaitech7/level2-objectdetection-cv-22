import pickle
import pandas as pd

# Pickle 파일 로드
with open('/data/ephemeral/home/mmdetectionv3/work_dirs/dino-5scale_swin-l_8xb2-12e_coco.pkl', 'rb') as file:
    predictions = pickle.load(file)

# 데이터 변환을 위한 리스트 초기화
prediction_strings = []
image_ids = []

# 예측 데이터를 변환하여 저장
for prediction in predictions:
    prediction_string = ''
    for bbox, score, label in zip(prediction['pred_instances']['bboxes'], 
                                  prediction['pred_instances']['scores'], 
                                  prediction['pred_instances']['labels']):
        prediction_string += f"{label.item()} {score.item()} {bbox[0].item()} {bbox[1].item()} {bbox[2].item()} {bbox[3].item()} "

    prediction_strings.append(prediction_string.strip())
    image_ids.append('test/' + prediction['img_path'].split('/')[-1]) 

# 데이터프레임 생성 및 CSV 저장
submission_df = pd.DataFrame({
    'PredictionString': prediction_strings,
    'image_id': image_ids
})

# CSV 파일 저장
submission_df.to_csv('/data/ephemeral/home/mmdetectionv3/work_dirs/dino-5scale_swin-l_8xb2-12e_coco/dino_k2_output.csv', index=False)