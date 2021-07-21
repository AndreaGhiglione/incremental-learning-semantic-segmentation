import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce

import models
from modules import DeeplabV3, BiSeNet


def make_model(opts, classes=None, is_old=False):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    if opts.context_path == 'resnet18':
        norm = nn.BatchNorm2d
        print('no InPlaceABNSync pretrained for resnet18 -> standard BatchNorm2d and torchvision resnet18 will be used')

    # fixed number of out channels for FFM
    out_channels = 32
    
    bisenet = BiSeNet(out_channels=out_channels, context_path=opts.context_path, pretrained_path=opts.pretrained_path, norm_act=norm, is_old=is_old)

    if classes is not None:
        model = IncrementalSegmentationModule(bisenet, out_channels, opts, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        model = SegmentationModule(bisenet, out_channels, opts.num_classes, opts.fusion_mode)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, bisenet, out_channels, opts, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationModule, self).__init__()
        self.bisenet = bisenet

        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        
        if opts.context_path == 'resnet18':
            self.sup1 = nn.ModuleList(
                [nn.Conv2d(256, c, 1) for c in classes]
            )
            self.sup2 = nn.ModuleList(
                [nn.Conv2d(512, c, 1) for c in classes]
            )
        else:
            self.sup1 = nn.ModuleList(
                [nn.Conv2d(1024, c, 1) for c in classes]
            )
            self.sup2 = nn.ModuleList(
                [nn.Conv2d(2048, c, 1) for c in classes]
            )
        
        self.cls = nn.ModuleList(
            [nn.Conv2d(out_channels, c, 1) for c in classes]
        )

        self.classes = classes
        self.out_channels = out_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False, phase='train'):
        bisenet_out = self.bisenet(x, phase=phase)
        if len(bisenet_out) == 3:
            x_bisenet, cx1, cx2 = bisenet_out
        else:
            x_bisenet = bisenet_out[0]

        if len(bisenet_out) == 3:
            out_sup1 = []
            for mod in self.sup1:
                out_sup1.append(mod(cx1))
            cx1_sup = torch.cat(out_sup1, dim=1)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=x.size()[-2:], mode='bilinear')

            out_sup2 = []
            for mod in self.sup2:
                out_sup2.append(mod(cx2))
            cx2_sup = torch.cat(out_sup2, dim=1)
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=x.size()[-2:], mode='bilinear')

        out = []
        for mod in self.cls:
            out.append(mod(x_bisenet))
        x_o = torch.cat(out, dim=1)

        if len(bisenet_out) == 3:
            if ret_intermediate:
                return x_o, cx1_sup, cx2_sup, x_bisenet
            return x_o, cx1_sup, cx2_sup
        else:
            if ret_intermediate:
                return x_o, x_bisenet
            return (x_o,)

    def init_new_supervision(self, device):
        sup1, sup2 = self.sup1[-1], self.sup2[-1]
        imprinting_w1, imprinting_w2 = self.sup1[0].weight[0], self.sup2[0].weight[0]
        bkg_bias1, bkg_bias2 = self.sup1[0].bias[0], self.sup2[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias1 = (bkg_bias1 - bias_diff)
        new_bias2 = (bkg_bias2 - bias_diff)

        sup1.weight.data.copy_(imprinting_w1)
        sup1.bias.data.copy_(new_bias1)

        sup2.weight.data.copy_(imprinting_w2)
        sup2.bias.data.copy_(new_bias2)

        self.sup1[0].bias[0].data.copy_(new_bias1.squeeze(0))
        self.sup2[0].bias[0].data.copy_(new_bias2.squeeze(0))
    
    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False, phase='train'):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate, phase=phase)

        sem_logits = out[0]

        # sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
        sem_logits = functional.interpolate(sem_logits, scale_factor=8, mode="bilinear") # moved from bisenet.py

        if len(out) >= 3:
            if ret_intermediate:
                return sem_logits, out[1], out[2], {"pre_logits": out[3]}
            return sem_logits, out[1], out[2], {}
        else:
            if ret_intermediate:
                return sem_logits, {"pre_logits": out[1]}
            return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
