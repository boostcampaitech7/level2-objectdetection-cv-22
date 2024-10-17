from stratified_group_kfold.train_detectron_fixed import train_model
from stratified_group_kfold.inference.inference_detectron_fixed_fold import test_model
from config.config_22 import Config22
from utils.filename_reader import get_last_line_of_record

def main():
    # 1. 실행할 파일 번호: 0은 학습, 1은 테스트
    num = 0

    # ──────────────────────────────────────────────────────────── 학습

    # original train data를 사용할지, 나눈 fold를 사용할지 여부를 정하여 사용할 fold 번호(0~4) 지정
    # -1인 경우 original 데이터 사용(validation 없음)
    fold_idx = 3

    # ──────────────────────────────────────────────────────────── 테스트
    # inference_detectron_fixed_fold에서 visualization 여부
    visualized = False
    # 추론 시 /data/ephemeral/home/outputs 아래 있는 디렉토리 이름 반드시 넣어주기
    inference_name = 'faster_rcnn_R_101_FPN_3x_10-17 12:24output_fold_3'
    get_last_line_of_record(inference_name)

    mycfg = Config22()


    if num == 0:
        train_model(mycfg, fold_idx=fold_idx)
    elif num == 1:
        test_model(mycfg, visualized=visualized, filename=inference_name)


if __name__ == "__main__":
    main()