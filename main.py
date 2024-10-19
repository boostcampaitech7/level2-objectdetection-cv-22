import os
import platform

from config.config_22 import Config22
from stratified_group_kfold.train_detectron_fixed import train_model
from stratified_group_kfold.inference.inference_detectron_fixed_fold import test_model

from utils.map_computer import compute_map
from utils.filename_reader import get_last_line_of_record
from utils.visualizer import load_json_results, load_annotations, visualize_detection


def main():
    os_name = platform.system()

    if os_name == 'Windows':
        root = 'C:/Users/yeyec/workspace/server2'
    else:
        root = '/data/ephemeral/home'

    os.makedirs(root + '/outputs', exist_ok=True)

    """
        1. 실행할 파일 번호: 
            ● 0 : 학습 
            ● 1 : 테스트
            ● 2 : 학습 + 테스트
            ● 3 : visualize(mAP 계산 선택)
    """

    num = 3

    # ──────────────────────────────────────────────────────────── 0/2 : 학습 설정

    """
        2-1. original train data를 사용할지 여부 지정
            ● -1인 경우 original 데이터 사용(validation 없음)
            ● 그 외 fold 번호(0~4) 지정
    """
    fold_idx = 3

    # ──────────────────────────────────────────────────────────── 1/2 : 테스트 설정
    """
        2-2. 추론 시 디렉토리 이름 지정
            ● 넣지 않으면 맨 마지막 실행한 train에 대한 inference 진행
            ● visualization 여부 설정
    """
    inference_name = ''
    visualized = False
    
    # ──────────────────────────────────────────────────────────── 3 : visualize 설정
    """
        2-3. train 데이터 시각화 디렉토리 지정
            ● mAP 계산이 필요하면(True) 이미 계산된 파일이 존재하면(False) : needToCompute

            ● train 데이터 전체 json 파일 경로 : train_json_path
            ● validation 데이터 json 파일 경로 : val_json_path
            ● mAP 데이터 csv 파일 경로 : map_file_path

            ● train 데이터셋 디렉토리 경로 : dataset_dir
            ● 시각화한 이미지 저정 디렉토리 경로 : output_dir
    """
    needToCompute = True

    train_json_path = root + '/dataset/train.json'
    val_json_path = root + '/outputs/val.bbox.json'
    output_csv_path = root + '/outputs/output_map.csv'

    dataset_dir = root + '/dataset/train/'
    output_dir = root + '/outputs/visualized_images/bad_map'
    

    


    # ──────────────────────────────────────────────────────────── 실행하는 부분

    mycfg = Config22()

    if num == 0:
        train_model(mycfg, fold_idx=fold_idx)
    elif num == 1:
        get_last_line_of_record(root, inference_name)
        test_model(mycfg, visualized=visualized, filename=inference_name)
    elif num == 2:
        train_model(mycfg, fold_idx=fold_idx)
        get_last_line_of_record(root, inference_name)
        test_model(mycfg, visualized=visualized, filename='')
    elif num == 3:
        if needToCompute:
            compute_map(train_json_path, val_json_path, output_csv_path)

        results = load_json_results(val_json_path)
        annotations, images_info = load_annotations(train_json_path)
        print(visualize_detection(results, annotations, images_info, dataset_dir, output_dir, output_csv_path))
    else:
        print("프로그램 종료")



if __name__ == "__main__":
    main()