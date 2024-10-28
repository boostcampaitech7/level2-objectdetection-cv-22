import numpy as np
from ensemble_boxes import *

from detectron2.data import detection_utils as utils


# 평균 ensemble
def ensemble_predictions(fold_outputs, score_thresh):
    num_detections = min([len(fold['scores']) for fold in fold_outputs])
    
    truncated_fold_outputs = [
        {
            'scores': fold['scores'][:num_detections],
            'boxes': fold['boxes'][:num_detections],
            'targets': fold['targets'][:num_detections]
        }
        for fold in fold_outputs
    ]

    final_scores = np.mean([fold['scores'] for fold in truncated_fold_outputs], axis=0)
    final_boxes = np.mean([fold['boxes'] for fold in truncated_fold_outputs], axis=0)
    final_targets = np.mean([fold['targets'] for fold in truncated_fold_outputs], axis=0)

    final_targets = np.round(final_targets).astype(int)
    
    keep = final_scores > score_thresh
    
    return final_targets[keep], final_boxes[keep], final_scores[keep]

