from stratified_group_kfold.train_detectron_fixed import train_model
from stratified_group_kfold.inference.inference_detectron_fixed_fold import test_model
from config.config_22 import Config22
from utils.filename_reader import get_last_line_of_record

def main():
    """
        1. 실행할 파일 번호: 0은 학습, 1은 테스트, 2는 둘 다 이어서
    """

    num = 0

    # ──────────────────────────────────────────────────────────── 학습

    """
        2-1. original train data를 사용할지 여부 지정
            ● -1인 경우 original 데이터 사용(validation 없음)
            ● 그 외 fold 번호(0~4) 지정
    """
    fold_idx = 3

    # ──────────────────────────────────────────────────────────── 테스트
    """
        2-2. 추론 시 디렉토리 이름 지정
            ● 넣지 않으면 맨 마지막 실행한 train에 대한 inference 진행
            ● visualization 여부 설정
    """
    inference_name = 'faster_rcnn_R_101_FPN_3x_10-17 12:24output_fold_3'
    get_last_line_of_record(inference_name)
    visualized = False

    mycfg = Config22()


    if num == 0:
        train_model(mycfg, fold_idx=fold_idx)
    elif num == 1:
        test_model(mycfg, visualized=visualized, filename=inference_name)
    elif num == 2:
        train_model(mycfg, fold_idx=fold_idx)
        test_model(mycfg, visualized=visualized, filename='')


if __name__ == "__main__":
    main()