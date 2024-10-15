import numpy as np

from detectron2.data import detection_utils as utils


# 평균 ensemble
def ensemble_predictions(fold_outputs, score_thresh):
    final_scores = np.mean([fold['scores'] for fold in fold_outputs], axis=0)
    final_boxes = np.mean([fold['boxes'] for fold in fold_outputs], axis=0)
    final_targets = np.mean([fold['targets'] for fold in fold_outputs], axis=0)

    final_targets = np.round(final_targets).astype(int)
    keep = final_scores > score_thresh
    
    return final_targets[keep], final_boxes[keep], final_scores[keep]

