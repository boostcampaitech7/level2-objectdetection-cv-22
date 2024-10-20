import os
import platform

from config.config_22 import Config22
from stratified_group_kfold.train_detectron_fixed import train_model
from stratified_group_kfold.inference.inference_detectron_fixed_fold import test_model

from utils.map_computer import compute_map
from utils.filename_reader import get_last_line_of_record
from utils.visualizer import load_json_results, load_annotations, visualize_detection
from utils.gt_finder import save_gt_and_detection_failure
from utils.map_computer_all import analyze_val_results


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
            ● 4 : 타겟 이미지 gt, 맞은 박스, 오탐지 박스 생성(3번 과정 선행 필요)
            ● 5 : 전체 map 계산(validation에 대한 json 파일 필요)
    """

    num = 4

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
            ● mAP 데이터 csv 파일 경로(needToCompute=True시 생성) : output_csv_path
            ● 오탐지, 오분류에 대한 바운딩 박스 크기 별 로그와 통계 저장(무족권 생성) : log_file_path

            ● train 데이터셋 디렉토리 경로 : dataset_dir
            ● 시각화한 이미지 저정 디렉토리 경로 : output_dir
    """
    needToCompute = False

    train_json_path = root + '/dataset/train.json'
    val_json_path = root + '/outputs/val.bbox.json'
    output_csv_path = root + '/outputs/output_map.csv'
    log_file_path = os.path.join(root, 'outputs', 'visualization_log.txt')
    log_stats_path = os.path.join(root, 'outputs', 'log_stats.txt')

    dataset_dir = root + '/dataset/train/'
    output_dir = root + '/outputs/visualized_images/bad_map'
    
    # ──────────────────────────────────────────────────────────── 4 : 1개 이미지 gt 시각화 설정
    """
        2-4. 시각화할 이미지 아이디 지정
            ● 필요한 파일
                - visualization_log.txt 파일 요구 -> 3번 선행 필요
            ● 생성되는 파일
                - /outputs/detection_failed/0000_detextion_failed.jpg
                - /outputs/correct_and_gt_bboxes/0000_correct_and_gt.jpg

            ● 추가 기능 : gt_finder.py 파일 내 
                - visualize_single_image 함수 주석 제거 시 /outputs/ground_truth/0000_gt.jpg 생성(gt만 확인하는 이미지)
                - visualize_correct_bboxes_single_image 함수 주석 제거 시 /outputs/correct_bboxes.0000_correct_bboxes.jpg 생성(correct만 확인하는 이미지)
    """
    image_id = 172

    # ──────────────────────────────────────────────────────────── 4 : 1개 이미지 gt 시각화 설정
    """
        2-5. validation.json 파일을 이용한 전체 mAP 계산
            ● 필요한 파일
                - validation에 사용된 annotation 파일 지정: gt_file
                - validation에 대한 json파일 지정 : pred_file
            ● 생성되는 파일
                - AP와 AR, 클래스별 최종 mAP 값이 저장되는 텍스트 파일 : output_file
    """

    # 필요한 파일
    gt_file = root + '/dataset/val_fold_3.json'
    pred_file = root + '/outputs/val.bbox.json'

    # 생성될 파일
    output_file = root + '/outputs/val_analysis.txt'

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
        print(visualize_detection(results, annotations, images_info, dataset_dir, output_dir, output_csv_path, log_file_path, log_stats_path))
    elif num == 4:
        save_gt_and_detection_failure(root, image_id)
    elif num == 5:
        analyze_val_results(gt_file, pred_file, output_file)
    else:
        print("프로그램 종료")



if __name__ == "__main__":
    main()