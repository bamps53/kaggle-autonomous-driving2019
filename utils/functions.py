import torch
import numpy as np
import cv2
import torch.nn as nn


def predict_batch(model, batch_images):
    batch_preds = torch.sigmoid(model(batch_images))
    return batch_preds.detach().cpu().numpy()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def depth_trans(x):
    return 1 / sigmoid(x) - 1

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _nms_np(heat, kernel=3):
    heat = torch.Tensor(heat)
    heat = heat.view(-1,1355,3384)
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return (heat * keep).numpy()