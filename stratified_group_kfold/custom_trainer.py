import os
import copy
import torch
import wandb

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import EventWriter
from detectron2.engine import hooks

def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [                                            # 데이터 증강 옵션
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    T.RandomBrightness(0.8, 1.8),
    T.RandomContrast(0.6, 1.3),
    #T.RandomRotation(angle=[-15, 15]),
    #T.RandomCrop(crop_type="relative_range", crop_size=(0.8, 0.8)),
    #T.RandomLighting(0.1),
]

    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        #######################################################
        # for obj in dataset_dict.pop('annotations')
        for obj in dataset_dict.pop('annotations', [])
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        #return build_detection_train_loader(cfg, mapper=MyMapper)
        return build_detection_train_loader(cfg)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        # WandbLogger의 trainer 설정
        wandb_logger = WandbLogger()
        wandb_logger.set_trainer(self)
        hooks_list.append(hooks.PeriodicWriter([wandb_logger], period=100))
        return hooks_list

        return COCOEvaluator(dataset_name, output_dir=output_folder)

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