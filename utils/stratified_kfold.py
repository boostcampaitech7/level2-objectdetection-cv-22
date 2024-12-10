import os
import json
import random
import numpy as np

from config.config_22 import Config22
from collections import Counter, defaultdict

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()

    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []

        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)

        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None

        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i

        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def save_fold_data(data, train_idx, val_idx, fold_idx, path_dataset):
    """
        stratified_group_kfold 적용하여 파일 저장
    """
    train_data = {'images': [], 'annotations': [], 'categories': data['categories']}
    val_data = {'images': [], 'annotations': [], 'categories': data['categories']}
    
    image_id_map = {}

    for img in data['images']:
        image_id_map[img['id']] = img

    # Train data 구성
    for idx in train_idx:

        ann = data['annotations'][idx]
        img_id = ann['image_id']
        train_data['annotations'].append(ann)

        if img_id not in [img['id'] for img in train_data['images']]:
            train_data['images'].append(image_id_map[img_id])
    
    # Val data 구성
    for idx in val_idx:

        ann = data['annotations'][idx]
        img_id = ann['image_id']
        val_data['annotations'].append(ann)

        if img_id not in [img['id'] for img in val_data['images']]:
            val_data['images'].append(image_id_map[img_id])

    # 각 fold 데이터 저장 
    train_json = os.path.join(path_dataset, f'{Config22.filename_fold_train}{fold_idx}.json')
    val_json = os.path.join(path_dataset, f'{Config22.filename_fold_val}{fold_idx}.json')
    
    with open(train_json, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    with open(val_json, 'w') as f:
        json.dump(val_data, f, indent=4)


# def get_distribution(y):
#     y_distr = Counter(y)
#     y_vals_sum = sum(y_distr.values())

#     return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]


# distrs = [get_distribution(y)]
# index = ['training set']
