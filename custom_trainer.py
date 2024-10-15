import os
import copy
import torch

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator

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

        return COCOEvaluator(dataset_name, output_dir=output_folder)