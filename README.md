# **재활용 품목 분류를 위한 Object Detection**

<p align="center">
<img width="1238" alt="image" src="https://github.com/user-attachments/assets/63629f53-5f6a-4736-b7b9-ec25ffd27aa2" width="90%" height="90%"/>
</p>

## 1. Competiton Info

### Overview

- 재활용 품목 분류를 위한 Object Detection
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- 이미지에서 재활용 쓰레기를 분류를 하기 위한 Object Detection 모델을 개발한다
- 문제 해결을 위한 데이터셋으로는 COCO 포맷의 Annotation 파일과 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 이미지가 제공된다

### Timeline

- 2024.10.02 ~ 2024.10.24

### Evaluation

- 평가지표: Test set의 mAP50
- Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True로 판단한다

## 2. Team Info

### MEMBERS

| <img src="https://github.com/user-attachments/assets/24d0bd6b-0742-486e-990a-9570e62096e5" width="200" height="160"/> | <img src="https://github.com/user-attachments/assets/5295a4ad-e426-4cd8-8bef-c64b53bccc4c" width="200" height="160"/> | <img src="https://github.com/user-attachments/assets/d30e8cae-a3ea-41cf-8947-0410776394f9" width="200" height="160"/> | <img src="https://github.com/user-attachments/assets/8a1d8b6a-f243-4b41-877d-78a48e6fde0f" width="200" height="160"/> | <img src="https://github.com/user-attachments/assets/864a519a-10ea-4f2a-b020-0f20b893f01a" width="200" height="160"/> |
| :---: | :---: | :---: | :---: | :---: |
| [김예진](https://github.com/yeyechu) | [배형준](https://github.com/BaeHyungJoon) | [송재현](https://github.com/mongsam2) | [이재효](https://github.com/jxxhyo) | [차성연](https://github.com/MICHAA4) |

### Project Objective and Direction

- **직접 찾은 근거, 공신력있는 근거 활용하기**
    - 모델을 선택하거나 하이퍼파라미터를 조정하는 데 있어서, ChatGPT에 의존하지 않고 임의로 파라미터를 바꿔가며 최적값을 찾아가지 않는다
    - 이전 실험의 결과를 분석하여 활용하거나 논문을 참고하는 등, 반드시 믿을 수 있는 근거를 가지고 가설을 세우고 실험을 설계한다
- **심층적 분석**
    - 수행한 실험에 대해 값만 확인하는 것이 아니, 결과를 시각화하고 면밀하게 분석하여 팀원들과 의견을 나눈다

### Team Component

- **EDA, 데이터 분석 및 유틸리티 배포** : 김예진, 이재효, 차성연
- **하이퍼파라미터 및 데이터 증강 실험** : 김예진, 배형준, 송재현
- **모델 실험 및 분석** : 이재효, 차성연

## 3. Data EDA


<img width="800" alt="result" src="https://github.com/user-attachments/assets/45327f7d-41fb-439d-b456-996c37221875">


- 카테고리별 수와 BBox 면적별 수를 확인함으로써 데이터가 상당히 불균형하다는 것을 알 수 있었다
- BBox 크기는 상당히 넓은 범위에 걸쳐 분포하고 비율은 대부분 4를 넘지 않는 것을 확인하였다

## 4. Modeling

### Model Description

- 1-stage : RetinaNet, Yolov11
- 2-stage : Faster-RCNN, Cascade-RCNN
- Vit 기반 모델 : DINO

### Modeling Result

| **Model** | **mAP@50 (Private)** |
| --- | --- |
| NMS_Ensemble (DINO) | **0.7142** |
| WBF_Ensemble (모든 모델) | 0.6185 |

<img width="500" alt="result" src="https://github.com/user-attachments/assets/a6c646c5-1be0-496f-9e93-b67b0e58a201">

## 5. Result

### Leader Board

Team name : CV_22조
<p align="center">
<img src="https://github.com/user-attachments/assets/fff1e2c8-5953-424b-8ae9-519f43481f05" width="70%" height="70%"/>
</p>

### Feedback

- 근거 기반 실험 설계와 심층적 분석을 통해 신뢰성 있는 실험을 수행하고 팀원들과 체계적으로 협업을 하였다
- 직접 데이터 증강 코드를 구현하여 커스텀 증강 시도해보면 좋았을 것 같다

## 6. How to Run

### Project File Structure

```python
├── detectron2 # detectron2 관련 파일
├── eda # 데이터 EDA 파일
├── mmdetection # mmdetection
    ├── dataset # 데이터셋 파일 + K-Fold json 파일
    ├── mmdetection # mmdetection 라이브러리 파일 및 모델 config 추가
    └── tools # 앙상블 및 csv 변환 파일
├── ultralytics # yolov11 라이브러리 및 모델 config 추가
├── utils # 성능 분석을 위한 유틸리티
└── main.py # detectron2 실행 main 파일
```

- 현재 레포지토리를 클론한다
    
    ```bash
    git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-22.git
    ```
    
- 아래 경로에 데이터셋을 다운받는다
    
    ```bash
    cd level2-objectdetection-cv-22/mmdetection/dataset
    ```
    

### Detectron2

- 아래의 경로로 이동한다
    
    ```bash
    cd level2-objectdetection-cv-22
    ```
    
- Detectron2 라이브러리를 다운로드한다
    
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    ```
    
- 학습, 테스트, 성능 분석 파일 추출 등 가이드라인에 따라 실행한다
    
    ```bash
    cd level2-objectdetection-cv-22/detectron2 && python main.py
    ```
    

### MMDetection

- 아래의 경로에서 실행한다
    
    ```bash
    cd level2-objectdetection-cv-22/mmdetection/
    ```
    
- 원하는 모델로 학습한다
    
    ```bash
    python tools/train.py custom_configs/{사용할 모델 config.py}
    ```
    
- 학습된 모델로 추론한 결과를 pickle 파일로 저장한다
    
    ```bash
    python tools/test.py custom_configs/{사용할 모델 config.py} work_dirs/{모델.pth} --out work_dirs/{사용한 모델 이름a.pkl}
    ```
    
- pickle 파일을 csv 파일로 변환하여 결과를 확인한다
    
    ```python
    python pikel_to_csv.py
    ```
    

### Ultiralytics (Yolov11)

- 아래의 경로로 이동한다
    
    ```bash
    cd level2-objectdetection-cv-22
    ```
    
- Ultiralytics 라이브러리 다운로드 받는다
    
    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```
    
- 원하는 모델로 학습, 추론 및 csv 파일을 추출한다
    
    ```bash
    cd level2-objectdetection-cv-22/Ultiralytics && python main.py
    ```
