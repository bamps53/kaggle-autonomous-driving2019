import numpy as np
from catalyst.dl.core.callback import MetricCallback
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import copy
from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool
from functools import partial


def TranslationDistance(p, g, abs_dist=False):
    dx = p['x'] - g['x']
    dy = p['y'] - g['y']
    dz = p['z'] - g['z']
    diff0 = (g['x']**2 + g['y']**2 + g['z']**2)**0.5
    diff1 = (dx**2 + dy**2 + dz**2)**0.5
    if abs_dist:
        diff = diff1
    else:
        diff = diff1/diff0
    return diff


def RotationDistance(p, g):
    true = [g['pitch'], g['yaw'], g['roll']]
    pred = [p['pitch'], p['yaw'], p['roll']]
    q1 = R.from_euler('xyz', true)
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)
    W = (acos(W)*360)/pi
    if W > 180:
        W = 360 - W
    return W


def calc_accuracy(result_flg, result_dist):
    result_flg = np.array(result_flg)
    result_dist = np.array(result_dist)
    all_accuracy = np.mean(result_dist)
    negative_accuracy = np.mean(result_dist[result_flg == 0])
    return all_accuracy, negative_accuracy


def get_score(th_idx, org_targets, org_predictions, n_gt, n_pr, keep_gt=False, random_confidence=False):
    targets = copy.deepcopy(org_targets)
    predictions = copy.deepcopy(org_predictions)
    thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    thre_tr_dist = thres_tr_list[th_idx]
    thre_ro_dist = thres_ro_list[th_idx]

    result_flg = []  # 1 for TP, 0 for FP
    result_dist = []
    result_rotate = []
    scores = []
    MAX_VAL = 10**10

    for target, pred in zip(targets, predictions):
        for pcar in sorted(pred, key=lambda x: -x['confidence']):
            # find nearest GT
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate(target):
                tr_dist = TranslationDistance(pcar, gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    min_ro_dist = RotationDistance(pcar, gcar)
                    min_idx = idx

            # set the result
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    target.pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            result_dist.append(int(min_tr_dist < thre_tr_dist))
            result_rotate.append(int(min_ro_dist < thre_ro_dist))
            scores.append(pcar['confidence'])

    if np.sum(result_flg) > 0:
        n_tp = np.sum(result_flg)
        recall = n_tp / n_gt
        precision = n_tp / n_pr

        n_tp_dist = np.sum(result_dist)
        recall_dist = n_tp_dist / n_gt
        precision_dist = n_tp_dist / n_pr

        n_tp_rotate = np.sum(result_rotate)
        recall_rotate = n_tp_rotate / n_gt
        precision_rotate = n_tp_rotate / n_pr

        if random_confidence:
            random_scores = np.random.rand(len(result_flg))
            ap = average_precision_score(result_flg, random_scores)
        else:
            ap = average_precision_score(result_flg, scores)
        score = ap * recall
        f1 = 2 * (recall * precision) / (recall + precision)
        f1_dist = 2 * (recall_dist * precision_dist) / \
            (recall_dist + precision_dist)
        f1_rotate = 2 * (recall_rotate * precision_rotate) / \
            (recall_rotate + precision_rotate)
    else:
        n_tp = 0
        recall = 0
        ap = 0
        score = 0
        f1 = 0
        f1_dist = 0
        f1_rotate = 0

    all_dist_accuracy, negative_dist_accuracy = calc_accuracy(
        result_flg, result_dist)
    all_rotate_accuracy, negative_rotate_accuracy = calc_accuracy(
        result_flg, result_rotate,)

    result = {}
    result['ap{}'.format(th_idx)] = ap
    result['f1_score_{}'.format(th_idx)] = f1
    result['f1_dist_{}'.format(th_idx)] = f1_dist
    result['f1_rotate_{}'.format(th_idx)] = f1_rotate
    result['n_tp{}'.format(th_idx)] = n_tp
    result['recall{}'.format(th_idx)] = recall
    result['map_score{}'.format(th_idx)] = score
    result['all_dist_accuracy{}'.format(th_idx)] = all_dist_accuracy
    result['negative_dist_accuracy{}'.format(th_idx)] = negative_dist_accuracy
    result['all_rotate_accuracy{}'.format(th_idx)] = all_rotate_accuracy
    result['negative_rotate_accuracy{}'.format(
        th_idx)] = negative_rotate_accuracy

    return result


def calc_map_score(targets, predictions, return_map_only=False, random_confidence=False):
    all_result = {}
    mean_result = {}
    # count num of ground truth
    n_gt = 0
    for i in range(len(targets)):
        n_gt += len(targets[i])
    all_result['n_gt'] = n_gt
    mean_result['n_gt'] = n_gt

    # count positive prediction
    n_pr = 0
    for i in range(len(predictions)):
        n_pr += len(predictions[i])
    all_result['n_pr'] = n_pr
    mean_result['n_pr'] = n_pr

    ap_list = []
    max_workers = 10
    p = Pool(processes=max_workers)
    results = p.map(
        partial(get_score, org_targets=targets, org_predictions=predictions,
                n_gt=n_gt, n_pr=n_pr, random_confidence=random_confidence),
        list(range(10))
    )

    mean_result['map_score'] = 0
    mean_result['f1_score'] = 0
    mean_result['f1_dist'] = 0
    mean_result['f1_rotate'] = 0
    mean_result['mean_ap'] = 0
    mean_result['mean_recall'] = 0
    mean_result['mean_all_dist_accuracy'] = 0
    mean_result['mean_all_rotate_accuracy'] = 0
    mean_result['mean_negative_dist_accuracy'] = 0
    mean_result['mean_negative_rotate_accuracy'] = 0

    for result in results:
        for k, v in result.items():
            if 'map_score' in k:
                mean_result['map_score'] += v / 10
            if 'ap' in k:
                mean_result['mean_ap'] += v / 10
            if 'f1_score' in k:
                mean_result['f1_score'] += v / 10
            if 'f1_dist' in k:
                mean_result['f1_dist'] += v / 10
            if 'f1_rotate' in k:
                mean_result['f1_rotate'] += v / 10
            if 'recall' in k:
                mean_result['mean_recall'] += v / 10
            if 'all_dist_accuracy' in k:
                mean_result['mean_all_dist_accuracy'] += v / 10
            if 'all_rotate_accuracy' in k:
                mean_result['mean_all_rotate_accuracy'] += v / 10
            if 'negative_dist_accuracy' in k:
                mean_result['mean_negative_dist_accuracy'] += v / 10
            if 'negative_rotate_accuracy' in k:
                mean_result['mean_negative_rotate_accuracy'] += v / 10
            all_result[k] = v

    return all_result, mean_result
