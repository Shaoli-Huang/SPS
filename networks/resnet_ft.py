import os
import sys
import numpy as np
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cv2

import torchvision
from torchvision import models
import pdb


class MidBlock(nn.Module):
    def __init__(self,sp,numswap,mid_dim=1024,num_class=200):
        super(MidBlock, self).__init__()
        self.mcls = nn.Linear(mid_dim, num_class)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv4_1 = nn.Sequential(nn.Conv2d(mid_dim, mid_dim, 1, 1), nn.ReLU())
        self.minp = sp[0]
        self.maxp = sp[1]
        self.numswap = numswap
    def swap(self,x):
        bs = x.size(0)
        lam = np.random.beta(1, 1)
        sp = self.minp + lam * (self.maxp - self.minp)
        rp = torch.from_numpy(np.random.permutation(x.size(0))).cuda()
        actidx = torch.rand(x.size(1))
        sidx = actidx < sp
        sidx.to(x.device)
        x[:,sidx] = x[rp[:,None],sidx]
        return x
    def forward(self, x):
        conv4_1 = self.conv4_1(x)
        pool4_1 = self.max_pool(conv4_1).view(conv4_1.size(0),-1)
        mlogits = []
        if self.training:
           for i in range(self.numswap):
               mslogit = self.mcls(self.swap(pool4_1.clone()))
               mlogits.append(mslogit)
        else:
            mlogits = self.mcls(pool4_1)
        return mlogits



class MidNet(nn.Module):
    def __init__(self,conf,mid_dim=1024):
        super(MidNet, self).__init__()
        self.numbranch = len(conf.sp)
        self.spsBranchs = nn.ModuleList()
        for sp in conf.sp:
            self.spsBranchs.append(MidBlock(sp,conf.numswap,mid_dim,conf.num_class))
    def forward(self,x):
        logits = []
        for i in range(self.numbranch):
            logits.append(self.spsBranchs[i](x))
        return logits



class ResNet(nn.Module):

    def __init__(self,conf):
        super(ResNet, self).__init__()
        basenet = eval('models.'+conf.netname)(pretrained=conf.pretrained)
        self.conv3 = nn.Sequential(*list(basenet.children())[:-4])
        self.conv4 = list(basenet.children())[-4]
        self.midlevel = False
        self.isdetach = True
        if 'midlevel' in conf:
            self.midlevel = conf.midlevel
        if 'isdetach' in conf:
            self.isdetach = isdetach

        mid_dim = 1024
        feadim = 2048
        if conf.netname in ['resnet18','resnet34']:
            mid_dim = 256
            feadim = 512

        if self.midlevel:
            self.midnet = MidNet(conf,mid_dim)
        self.conv5 = list(basenet.children())[-3]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feadim, conf.num_class)

    def set_detach(self,isdetach=True):
        self.isdetach = isdetach


    def forward(self, x):
        x = self.conv3(x)
        conv4 = self.conv4(x)
        x = self.conv5(conv4)
        fea_pool = self.avg_pool(x).view(x.size(0), -1)
        logits = self.classifier(fea_pool)

        if self.midlevel:
            if self.isdetach:
                conv4_1 = conv4.detach()
            else:
                conv4_1 = conv4
            mlogits = self.midnet(conv4_1)
        else:
            mlogits = None

        return logits,x.detach(),mlogits


    def _init_weight(self, block):
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_params(self, param_name):
        ftlayer_params = list(self.conv3.parameters()) +\
                           list(self.conv4.parameters()) +\
                           list(self.conv5.parameters())
        ftlayer_params_ids = list(map(id, ftlayer_params))
        freshlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())

        return eval(param_name+'_params')


def get_net(conf):
    return ResNet(conf)
