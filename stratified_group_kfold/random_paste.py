import numpy as np
import cv2
import random
from detectron2.data.transforms import Augmentation
from detectron2.data.transforms.augmentation_impl import ResizeShortestEdge
from detectron2.structures import BoxMode
from detectron2.data.transforms.transform import NoOpTransform

class RandomPaste(Augmentation):
    def __init__(self, paste_image, paste_bboxes, p=0.5):
        """
        paste_image: 붙여넣을 이미지를 numpy 배열로 받음.
        paste_bboxes: 붙여넣을 이미지의 바운딩 박스들 [(xmin, ymin, xmax, ymax)] 형태
        p: 적용 확률
        """
        self.paste_image = paste_image
        self.paste_bboxes = paste_bboxes
        self.p = p

    def get_transform(self, img):
        if random.random() < self.p:
            
            img_h, img_w = img.shape[:2]
            paste_h, paste_w = self.paste_image.shape[:2]
            
            
            x_offset = random.randint(0, img_w - paste_w)
            y_offset = random.randint(0, img_h - paste_h)
            
            img[y_offset:y_offset+paste_h, x_offset:x_offset+paste_w] = self.paste_image

            
            paste_bboxes = []
            for bbox in self.paste_bboxes:
                xmin, ymin, xmax, ymax = bbox
                new_bbox = [
                    xmin + x_offset,
                    ymin + y_offset,
                    xmax + x_offset,
                    ymax + y_offset,
                ]
                paste_bboxes.append(new_bbox)

            return RandomPasteTransform(paste_bboxes)
        else:
            return NoOpTransform()


class RandomPasteTransform:
    def __init__(self, paste_bboxes):
        self.paste_bboxes = paste_bboxes

    def apply_image(self, img):
        return img

    def apply_coords(self, coords):
        return coords

    def apply_box(self, boxes):
        if isinstance(boxes, np.ndarray):
            return np.vstack([boxes, np.array(self.paste_bboxes)])
        else:
            return boxes
