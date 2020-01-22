from catalyst.dl.utils import criterion
from torch.nn.modules.loss import _Loss
from functools import partial
from catalyst.contrib.criterion import FocalLossBinary
from catalyst.dl.callbacks import CriterionCallback, CriterionAggregatorCallback, MixupCallback
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import numpy as np
from functools import partial


class HeatMapLoss(nn.Module):
    def __init__(self, eps=1e-12, focal=False, reduce='sum'):
        super().__init__()
        self.eps = eps
        self.focal = focal
        self.reduce = reduce

    def forward(self, prediction, mask_regr):
        mask = mask_regr[:, 0]
        pred_mask = torch.sigmoid(prediction[:, 0])

        # Binary mask loss
        if self.focal:
            mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (
                1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
        else:
            mask_loss = mask * \
                torch.log(pred_mask + 1e-12) + (1 - mask) * \
                torch.log(1 - pred_mask + 1e-12)

        if self.reduce == 'sum':
            mask_loss = -mask_loss.mean(0).sum()
        elif self.reduce == 'mean':
            mask_loss = -mask_loss.mean()
        elif self.reduce == 'num_point':
            mask_loss = -(mask_loss / mask.sum(1).sum(1)).mean(0).sum()
        return mask_loss


def depth_transform(x):
    return 1 / torch.sigmoid(x) - 1


class ZLoss(nn.Module):
    def __init__(self, z_pos):
        super().__init__()
        self.z_pos = z_pos

    def forward(self, prediction, mask_regr):
        mask = mask_regr[:, 0]
        regr = mask_regr[:, self.z_pos]
        pred_mask = torch.sigmoid(prediction[:, 0])
        pred_regr = prediction[:, self.z_pos]
        pred_regr = depth_transform(pred_regr)

        # Regression L1 loss
        regr_loss = (torch.abs(pred_regr - regr).sum(1) *
                     mask).sum(1).sum(1) / mask.sum(1).sum(1)
        regr_loss = regr_loss.mean(0)
        return regr_loss


class PitchLoss(nn.Module):
    def __init__(self, pitch_pos):
        super().__init__()
        self.pitch_pos = pitch_pos

    def forward(self, prediction, mask_regr):
        mask = mask_regr[:, 0]
        regr = mask_regr[:, self.pitch_pos]
        pred_mask = torch.sigmoid(prediction[:, 0])
        pred_regr = prediction[:, self.pitch_pos]

        # Regression L1 loss
        regr_loss = (torch.abs(pred_regr - regr).sum(1) *
                     mask).sum(1).sum(1) / mask.sum(1).sum(1)
        regr_loss = regr_loss.mean(0)
        return regr_loss


def get_criterion_and_callback(config):
    if config.train.mixup:
        print('info: turn on mixup')
        CC = MixupCallback
    else:
        CC = CriterionCallback

    if config.loss.name == 'MaskDepthPitch':
        criterion = {
            "heatmap": HeatMapLoss(
                focal=config.loss.params.focal,
                reduce=config.loss.params.reduce,
            ),
            "Z": ZLoss(z_pos=config.data.z_pos),
            "pitch": PitchLoss(pitch_pos=config.data.pitch_pos)
        }
        callbacks = [
            # Each criterion is calculated separately.
            CC(
                input_key="targets",
                prefix="loss_heatmap",
                criterion_key="heatmap"
            ),
            CC(
                input_key="targets",
                prefix="loss_z",
                criterion_key="Z",
            ),
            CC(
                input_key="targets",
                prefix="loss_pitch",
                criterion_key="pitch",
            ),
            # And only then we aggregate everything into one loss.
            CriterionAggregatorCallback(
                prefix="loss",
                loss_aggregate_fn="weighted_sum",
                loss_keys={
                    "loss_heatmap": config.loss.params.heatmap_weight,
                    "loss_z": config.loss.params.z_weight,
                    "loss_pitch": config.loss.params.pitch_weight
                },
            )
        ]

    else:
        criterion = None
        callbacks = None

    return criterion, callbacks
