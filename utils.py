import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import nn
import torch.nn.functional as F


def draw_trajectory_on_ax(ax: Axes, trajectories, confs, line_type='o-', transparent=True, xlim=(-30, 30), ylim=(0, 100)):
    '''
    ax: matplotlib.axes.Axes, the axis to draw trajectories on
    trajectories: List of numpy arrays of shape (num_points, 2 or 3)
    confs: List of numbers, 1 means gt
    '''

    for idx, (trajectory, conf) in enumerate(zip(trajectories, confs)):
        label = 'gt' if conf == 1 else 'pred%d (%.3f)' % (idx, conf)
        alpha = np.clip(conf, 0.1, None) if transparent else 1.0
        ax.plot(-trajectory[:, 1], trajectory[:, 0], line_type, label=label, alpha=alpha)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()

    return ax


def get_val_metric(pred_cls, pred_trajectory, labels, namespace='val'):
    rtn_dict = dict()
    bs, M, num_pts, _ = pred_trajectory.shape

    # Lagecy metric: Prediction L2 loss
    pred_label = torch.argmax(pred_cls, -1)  # B,
    pred_trajectory_single = pred_trajectory[torch.tensor(range(bs), device=pred_cls.device), pred_label, ...]
    l2_dists = F.mse_loss(pred_trajectory_single, labels, reduction='none')  # B, num_pts, 2 or 3

    # Lagecy metric: cls Acc
    gt_trajectory_M = labels[:, None, ...].expand(-1, M, -1, -1)
    l2_distances = F.mse_loss(pred_trajectory, gt_trajectory_M, reduction='none').sum(dim=(2, 3))  # B, M
    best_match = torch.argmin(l2_distances, -1)  # B,
    rtn_dict.update({'l2_dist': l2_dists.mean(dim=(1, 2)), 'cls_acc': best_match == pred_label})

    # New Metric
    distance_splits = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 1000))
    AP_thresholds = (0.5, 1, 2)
    euclidean_distances = l2_dists.sum(-1).sqrt()  # euclidean distances over the points: [B, num_pts]    
    x_distances = labels[..., 0]  # B, num_pts

    for min_dst, max_dst in distance_splits:
        points_mask = (x_distances >= min_dst) & (x_distances < max_dst)  # B, num_pts,
        if points_mask.sum() == 0:
            continue  # No gt points in this range
        rtn_dict.update({'eucliden_%d_%d' % (min_dst, max_dst): euclidean_distances[points_mask]})  # [sum(mask), ]

        for AP_threshold in AP_thresholds:
            hit_mask = (euclidean_distances < AP_threshold) & points_mask
            rtn_dict.update({'AP_%d_%d_%s' % (min_dst, max_dst, AP_threshold): hit_mask[points_mask]})

    # add namespace
    if namespace is not None:
        for k in list(rtn_dict.keys()):
            rtn_dict['%s/%s' % (namespace, k)] = rtn_dict.pop(k)
    return rtn_dict