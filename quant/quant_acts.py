# This file is part of PWLQ repository.
# Copyright (c) Samsung Semiconductor, Inc.
# All rights reserved.

import copy
import torch
import torch.nn as nn

from .pwlq import *
from .uniform import *

##########################################################################################
####  Quantization of Activations 
##########################################################################################

class QuantAct(nn.Module):
    '''
    Quantize actications including:
    (1) the input of conv layer
    (2) the input of linear fc layer
    (3) the input of pooling layer
    '''
    def __init__(self, act_bits, get_stats, minv=None, maxv=None, 
        cali_sample_size=512, cali_batch_size=4, topk=10):
        '''
        cali_sample_size: calibration sample size, typically from random training data
        cali_batch_size: calibration sampling batch size
        topk: calibrate topk lower and upper bounds
        '''
        super(QuantAct, self).__init__()
        self.act_bits = act_bits
        self.get_stats = get_stats
        self.index = 0
        self.topk = topk
        self.sample_batches = cali_sample_size // cali_batch_size
        stats_size = (self.sample_batches, self.topk) if self.get_stats else 1
        self.register_buffer('minv', torch.zeros(stats_size))
        self.register_buffer('maxv', torch.zeros(stats_size))

    def forward(self, x):
        if self.get_stats:
            y = x.clone()
            y = torch.reshape(y, (-1,))
            y, indices = torch.sort(y)
            topk_mins = y[:self.topk]
            topk_maxs = y[-self.topk:]
            if self.index < self.sample_batches:
                self.minv[self.index, :] = topk_mins
                self.maxv[self.index, :] = topk_maxs
                self.index += 1

        if self.act_bits > 0:
            ## uniform quantization
            if self.minv is not None:
                if self.minv >= 0.0: # activation after relu
                    self.minv *= 0.0
                    self.signed = False
                else: 
                    self.maxv = max(-self.minv, self.maxv) 
                    self.minv = - self.maxv
                    self.signed = True
            x = uniform_symmetric_quantizer(x, bits=self.act_bits, 
                    minv=self.minv, maxv=self.maxv, signed=self.signed)
        return x


def quant_model_acts(model, act_bits, get_stats, cali_batch_size=4):
    """
    Add quantization of activations for a pretrained model recursively
    """
    if type(model) in [nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d]:
        quant_act = QuantAct(act_bits, get_stats, cali_batch_size=cali_batch_size)
        return nn.Sequential(quant_act, model)
    elif type(model) == nn.Sequential:
        modules = []
        for name, module in model.named_children():
            modules.append(quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size))
        return nn.Sequential(*modules)
    else:
        quantized_model = copy.deepcopy(model)
        for attribute in dir(model):
            module = getattr(model, attribute)
            if isinstance(module, nn.Module):
                setattr(quantized_model, attribute, 
                    quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size))
        return quantized_model


def save_model_act_stats(model, save_path):
    checkpoint = model.state_dict()
    act_stats = copy.deepcopy(checkpoint)
    for key in checkpoint:
        if '.minv' not in key and '.maxv' not in key:
            del act_stats[key]
    torch.save(act_stats, save_path)
    return act_stats


def load_model_act_stats(model, load_path, act_clip_method):
    checkpoint = model.state_dict()
    act_stats = torch.load(load_path)
    for key in act_stats:
        min_or_max = 'min' if '.minv' in key else 'max'
        value = act_clip_bounds(act_stats[key], act_clip_method, min_or_max)
        key = key.replace('module.', '')
        checkpoint[key][0] = value
    model.load_state_dict(checkpoint)
    return model


def act_clip_bounds(stats, act_clip_method, min_or_max):
    if act_clip_method.startswith('top'):
        topk = int(act_clip_method.split('_')[1])
        assert(topk <= 20)
        stats = stats[:, :topk] if min_or_max == 'min' else stats[:, -topk:]
        values, indices = torch.median(stats, 1)
        return torch.mean(values)
    elif act_clip_method.startswith('clip'):
        clip_coef = float(act_clip_method.split('_')[1])
        clip_value = torch.min(stats) if min_or_max == 'min' else torch.max(stats)
        return clip_coef * clip_value
    else:
        raise RuntimeError("Please implement for activation clip method: %s !!!" % act_clip_method) 
