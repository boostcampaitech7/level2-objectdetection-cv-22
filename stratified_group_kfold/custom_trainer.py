import os
import copy
import torch
import torch.nn as nn
import wandb
import random

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import EventWriter
from detectron2.engine import hooks

#from random_paste import RandomPaste, RandomPasteTransform

def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [                                            # 데이터 증강 옵션
    #T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    #T.RandomBrightness(0.8, 1.8),
    #T.RandomContrast(0.6, 1.3),
    #T.RandomRotation(angle=[-15, 15]),
    #T.RandomCrop(crop_type="relative_range", crop_size=(random.uniform(0.5, 0.9), random.uniform(0.5, 0.9))),
    #T.RandomLighting(0.1),
    T.ResizeShortestEdge(short_edge_length=(1200, 1600), max_size=2048),
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations', [])
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict


def weighted_loss(pred_boxes, target_boxes):
    
    smooth_l1_loss = nn.SmoothL1Loss()
    loss = smooth_l1_loss(pred_boxes, target_boxes)
    
    # 작은 박스에 대해 가중치 부여
    box_widths = target_boxes[:, 2] - target_boxes[:, 0]
    weights = 1.0 / box_widths

    # 가중치를 각 손실 값에 적용
    weighted_loss = loss * weights.unsqueeze(1)
    return weighted_loss.mean()


def extract_random_patch(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max]


# def random_paste_mapper(dataset_dict):
#     dataset_dict = copy.deepcopy(dataset_dict)
#     image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
#     transform_list = [                                            # 데이터 증강 옵션
#     T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
#     T.RandomBrightness(0.8, 1.8),
#     T.RandomContrast(0.6, 1.3),
#     #T.RandomRotation(angle=[-15, 15]),
#     #T.RandomCrop(crop_type="relative_range", crop_size=(0.8, 0.8)),
#     #T.RandomLighting(0.1),
# ]

#     image, transforms = T.apply_transform_gens(transform_list, image)
#     annos = dataset_dict.get('annotations', [])

#     if len(annos) > 0:
#         # 바운딩 박스 선택 (x_min, y_min, x_max, y_max)
#         selected_anno = random.choice(annos)
#         bbox = selected_anno['bbox']
#         x_min, y_min, x_max, y_max = map(int, bbox)

#         paste_image = image[y_min:y_max, x_min:x_max]

#         # 붙여넣기 위치를 랜덤하게 선택 (이미지 안에서)
#         img_h, img_w = image.shape[:2]
#         paste_h, paste_w = paste_image.shape[:2]
#         x_offset = random.randint(0, img_w - paste_w)
#         y_offset = random.randint(0, img_h - paste_h)

#         # 잘라낸 객체를 원본 이미지에 붙여넣기
#         image[y_offset:y_offset+paste_h, x_offset:x_offset+paste_w] = paste_image

#         # 붙여넣어진 객체의 새로운 바운딩 박스 생성
#         new_bbox = [x_offset, y_offset, x_offset + paste_w, y_offset + paste_h]
#         dataset_dict['annotations'].append({
#             'bbox': new_bbox,
#             'bbox_mode': selected_anno['bbox_mode'],
#             'category_id': selected_anno['category_id']
#         })

#     random_paste = RandomPaste(paste_image, paste_bboxes, p=0.5)
#     paste_transform = random_paste.get_transform(image)

    
#     dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dict.pop('annotations', [])
#         if obj.get('iscrowd', 0) == 0
#     ]
    
#     instances = utils.annotations_to_instances(annos, image.shape[:2])
#     dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
#     return dataset_dict


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyMapper)
        #return build_detection_train_loader(cfg)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)

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
            #wandb.log(logging.getLogger(__name__))

    def set_trainer(self, trainer):
        # trainer 객체 설정
        self.trainer = trainer

    def after_step(self):
        self.write()

    def after_train(self):
        self.write()