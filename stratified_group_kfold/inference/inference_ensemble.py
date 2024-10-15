import numpy as np

from detectron2.data import detection_utils as utils


# ensemble 방식 선택(최상위)
def ensemble_predictions(method, fold_outputs, score_thresh):
    if method == 'mean':
        ensemble_predictions_mean(fold_outputs, score_thresh)
    elif method == 'major':
        ensemble_predictions_major(fold_outputs, score_thresh)


# 평균 ensemble
def ensemble_predictions_mean(fold_outputs, score_thresh):
    final_scores = np.mean([fold['scores'] for fold in fold_outputs], axis=0)
    final_boxes = np.mean([fold['boxes'] for fold in fold_outputs], axis=0)
    final_targets = np.mean([fold['targets'] for fold in fold_outputs], axis=0)

    final_targets = np.round(final_targets).astype(int)
    keep = final_scores > score_thresh
    
    return final_targets[keep], final_boxes[keep], final_scores[keep]


# 다수결 방식 ensemble
def ensemble_predictions_major(fold_outputs, score_thresh):
    
    all_boxes = np.vstack([fold['boxes'] for fold in fold_outputs])
    all_scores = np.hstack([fold['scores'] for fold in fold_outputs])
    all_targets = np.hstack([fold['targets'] for fold in fold_outputs])
    
    
    keep = utils.nms(all_boxes, all_scores, score_thresh)
    
    final_boxes = all_boxes[keep]
    final_scores = all_scores[keep]
    
    final_targets = np.array([np.bincount(fold['targets']).argmax() for fold in fold_outputs])
    
    return final_targets, final_boxes, final_scores