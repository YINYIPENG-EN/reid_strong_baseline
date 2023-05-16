import torch
import torch.nn as nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a

import numpy as np
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, model_name, neck, neck_feat):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(BasicBlock, [2, 2, 2, 2], last_stride=last_stride)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(BasicBlock, [3, 4, 6, 3], last_stride=last_stride)
        elif model_name == 'resnet50':
            self.base = ResNet(Bottleneck, [3, 4, 6, 3], last_stride=last_stride)
        elif model_name == 'resnet101':
            self.base = ResNet(Bottleneck, [3, 4, 23, 3], last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        model = self.base
        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict, strict=False)
        del pretrained_dict
        print(f"{model_name} loading pretrained model weight...")

        # add avgpool layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        # add fc layer
        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        global_feat = self.gap(x)
        # reshape
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(global_feat)
            return cls_score, global_feat
        else:
            # 测试的时候抛弃全连接
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
