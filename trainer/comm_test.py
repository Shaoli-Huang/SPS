import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import time
import logging

from utils import *

def validate(train_loader, model, criterion, conf):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    scores = AverageAccMeter()
    mscores = [AverageAccMeter() for i in range(len(conf.sp))]
    ascores = AverageAccMeter()
    end = time.time()
    model.eval()

    time_start = time.time()
    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader))

    for idx, (input, target) in enumerate(pbar):
        # measure data loading time
        data_time.add(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        if 'inception' in conf.netname:
            output = model(input)
        else:
            output,_,moutput = model(input)
        scores.add(output.data, target)
        if 'midlevel' in conf:
            if conf.midlevel:
                allout = output.data
                for i in range(len(conf.sp)):
                    mscores[i].add(moutput[i].data, target)
                    allout += moutput[i].data
                ascores.add(allout, target)

        loss = torch.mean(criterion(output, target))
        losses.add(loss.item(), input.size(0))
        del loss,output

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()
        pbar.set_postfix(batch_time=batch_time.value(), data_time=data_time.value(), loss=losses.value())
        msvalue = [ms.value() for ms in mscores]

    return scores.value(), losses.value(),msvalue,ascores.value()
