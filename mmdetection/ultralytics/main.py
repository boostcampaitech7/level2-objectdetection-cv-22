from ultralytics import YOLO
import glob
import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader, Dataset
import cv2

# 모델 로드: YOLO 클래스의 인스턴스를 생성하고, 구성 파일과 가중치를 로드합니다.
model = YOLO("../ultralytics/cfg/models/11/yolo11-trash.yaml").load("yolo11n.pt")

# 모델 훈련: 지정된 데이터셋으로 모델을 훈련합니다.
train_results = model.train(
    data="../../datasets/trash.yaml",  # 데이터셋 경로
    epochs=1,  # 훈련 에포크 수
    imgsz=1024,  # 이미지 크기
    device="0",  # GPU 장치 번호
)

# 모델 검증: 훈련된 모델을 검증하고, mAP 지표를 출력합니다.
metrics = model.val()  
print(metrics.box.map)  # 전체 mAP
print(metrics.box.map50)  # IoU 50%에서의 mAP
print(metrics.box.map75)  # IoU 75%에서의 mAP
print(metrics.box.maps)  # 각 클래스별 mAP

# 테스트 이미지 경로 설정
test_images_path = "../../home/dataset/test/*.jpg"

# 테스트 이미지에 대한 예측 수행
results = model(test_images_path)
formatted_results = []

# 각 이미지에 대한 예측 결과를 포맷팅
for result in results:
    image_id = result.path.split("/")[-1] + ".jpg"  # 이미지 파일명 추출
    prediction_string = []

    # 각 바운딩 박스에 대해 클래스, 신뢰도, 좌표를 포맷팅
    for box in result.boxes:
        label = int(box.cls[0])  # 클래스 레이블
        score = box.conf[0]  # 신뢰도 점수
        x_center, y_center, width, height = box.xywh[0]  # 바운딩 박스 중심 및 크기
        xmin = x_center - width / 2  # 좌측 상단 x 좌표
        ymin = y_center - height / 2  # 좌측 상단 y 좌표
        xmax = x_center + width / 2  # 우측 하단 x 좌표
        ymax = y_center + height / 2  # 우측 하단 y 좌표
        prediction_string.append(f"{label} {score:.6f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}")

    prediction_string = " ".join(prediction_string)
    formatted_results.append({"PredictionString": prediction_string, "image_id": image_id})

# 결과를 데이터프레임으로 변환
df = pd.DataFrame(formatted_results)

# 결과를 CSV 파일로 저장
df.to_csv("/runs/detect/yolo11.csv", index=False)
