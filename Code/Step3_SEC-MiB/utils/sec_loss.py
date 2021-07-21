import torch
import numpy as np
from scipy.ndimage import zoom

from krahenbuhl2013 import CRF

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

EPS_PROB = 1e-4

def softmax_layer(preds):
    preds = preds
    pred_max, _ = torch.max(preds, dim=1, keepdim=True)
    pred_exp = torch.exp(preds - pred_max.clone().detach())
    probs = pred_exp / torch.sum(pred_exp, dim=1, keepdim=True) + EPS_PROB
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs

def seed_loss_layer(probs, gt_map):
    cnt = torch.sum(gt_map, dim=[1, 2, 3], keepdim=True)
    loss = -torch.mean(torch.sum(gt_map * torch.log(probs), dim=[1, 2, 3], keepdim=True) / cnt)
    return loss

def expand_loss_layer(probs, labels, num_classes):
    stat = labels[:, :, :, 1:]
    probs_bkg = probs[:, 0, :, :]
    probs = probs[:, 1:, :, :]

    probs_max, _ = torch.max(torch.max(probs, dim=3)[0], dim=2)

    q_fg = 0.996 # d+ prior
    probs_sort, _ = torch.sort(probs.contiguous().view(-1, num_classes, 41 * 41), dim=2)
    weights = probs_sort.new_tensor([q_fg ** i for i in range(41 * 41 -1, -1, -1)])[None, None, :]
    z_fg = torch.sum(weights)
    probs_mean = torch.sum((probs_sort * weights) / z_fg, dim=2)

    q_bkg = 0.999 # dbkg prior
    probs_bkg_sort, _ = torch.sort(probs_bkg.contiguous().view(-1, 41 * 41), dim=1)
    weights_bkg = probs_sort.new_tensor([q_bkg ** i for i in range(41 * 41 -1, -1, -1)])[None, :]
    z_bkg = torch.sum(weights_bkg)
    probs_bkg_mean = torch.sum((probs_bkg_sort * weights_bkg) / z_bkg, dim=1)

    stat_2d = (stat[:, 0, 0, :] > 0.5).float()
    loss_1 = -torch.mean(torch.sum((stat_2d * torch.log(probs_mean) / (torch.sum(stat_2d, dim=1, keepdim=True) + 1)), dim=1))
    loss_2 = -torch.mean(torch.sum(((1 - stat_2d) * torch.log(1 - probs_max) / torch.sum(1 - stat_2d, dim=1, keepdim=True)), dim=1))
    loss_3 = -torch.mean(torch.log(probs_bkg_mean))

    loss = loss_1 + loss_2 + loss_3
    return loss

def crf_layer(out, images, iternum):
    unary = np.transpose(np.array(out.cpu().clone().data), [0, 2, 3, 1])
    mean_pixel = np.array([104.0, 117.0, 123.0])
    img = images.cpu().data
    img = zoom(img, (1, 1, 41 / img.shape[2], 41 / img.shape[3]), order=1)

    img = img + mean_pixel[None, :, None, None]
    img = np.transpose(np.round(img), [0, 2, 3, 1])

    N = unary.shape[0]
    result = np.zeros(unary.shape)

    for i in range(N):
        result[i] = CRF(img[i], unary[i], maxiter=iternum, scale_factor=12.0)
    result = np.transpose(result, [0, 3, 1, 2])
    result[result < EPS_PROB] = EPS_PROB
    result = result / np.sum(result, axis=1, keepdims=True)

    return np.log(result)

def constrain_loss_layer(probs, probs_smooth_log):
    probs_smooth = torch.exp(probs.new_tensor(probs_smooth_log, requires_grad=True))
    loss = torch.mean(torch.sum(probs_smooth * torch.log(probs_smooth / probs), dim=1))
    return loss