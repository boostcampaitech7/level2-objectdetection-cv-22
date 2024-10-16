import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import hooks
from detectron2.utils.events import EventWriter

import wandb


# Register Dataset
try:
    register_coco_instances('coco_trash_train', {}, '../../dataset/train_fold_0.json', '../../dataset/')
except AssertionError:
    pass

try:
    register_coco_instances('coco_trash_test', {}, '../../dataset/val_fold_0.json', '../../dataset/')
except AssertionError:
    pass

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


# config 불러오기
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

# Project Settings
project_name = "faster_rcnn_base"
name  = "test"
wandb.init(project=project_name, config=cfg.clone(), name=name)


#----------------------------------------------------------------------------------------------------------------
# config 수정하기
cfg.DATASETS.TRAIN = ('coco_trash_train',)
cfg.DATASETS.TEST = ('coco_trash_test',)

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (8000,12000)
cfg.SOLVER.GAMMA = 0.005
cfg.SOLVER.CHECKPOINT_PERIOD = 3000

cfg.OUTPUT_DIR = f'./{project_name}'

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

cfg.TEST.EVAL_PERIOD = 100
#----------------------------------------------------------------------------------------------------------------


# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
import detectron2.data.transforms as T

def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict


# trainer - DefaultTrainer를 상속
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = MyMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok = True)
            output_folder = './output_eval'
            
        return COCOEvaluator(
            dataset_name=dataset_name, 
            tasks=["bbox"],  # 바운딩 박스 task를 평가
            distributed=False,  # 분산 학습이 아닌 경우 False
            output_dir=output_folder
        )
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        # WandbLogger의 trainer 설정
        wandb_logger = WandbLogger()
        wandb_logger.set_trainer(self)
        hooks_list.append(hooks.PeriodicWriter([wandb_logger], period=100))
        return hooks_list


# wandb logger
class WandbLogger(EventWriter):
    def __init__(self):
        super().__init__()
        self.trainer = None  # trainer를 저장할 변수 초기화

    def write(self):
        # 학습 중 발생하는 메트릭을 Wandb에 기록
        if self.trainer is not None:
            # 모든 메트릭을 가져와서 Wandb에 로그
            metrics = {k: v.median(20) for k, v in self.trainer.storage.histories().items()}
            wandb.log(metrics)

    def set_trainer(self, trainer):
        # trainer 객체 설정
        self.trainer = trainer

    def after_step(self):
        self.write()

    def after_train(self):
        self.write()

os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


