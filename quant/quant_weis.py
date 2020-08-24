# This file is part of PWLQ repository.
# Copyright (c) Samsung Semiconductor, Inc.
# All rights reserved.

import torch
import numpy as np

from .pwlq import *
from .uniform import *


##########################################################################################
####  Quantization of Weights 
##########################################################################################

def quant_checkpoint(checkpoint, weight_layers, args):
    '''
    Quantize the weights per-channel per-layer for a pre-trained checkpoint
    '''
    bits = args.wei_bits
    scale_bits = args.scale_bits
    bias_corr = args.bias_corr
    break_point_option = args.break_point

    print('quantizing weights into %s bits, %s layers' % (bits, len(weight_layers)))

    if bits == 0:
        return checkpoint, 0

    all_quant_error, all_quant_num = 0, 0
    all_tail_num = 0
    for each_layer in sorted(weight_layers):
        
        each_layer_weights = checkpoint[each_layer].clone()

        print('quantize for: %s, size: %s' % (each_layer, each_layer_weights.size()))
        print('weights range: (%.4f, %.4f)' % 
                            (torch.min(each_layer_weights), torch.max(each_layer_weights)))

        quant_error, quant_num, layer_tail_num = 0, 0, 0
        output_channel_num = each_layer_weights.size()[0]
        # channel-wise quant for each output channel
        for c in range(output_channel_num):  
            w = each_layer_weights[c, :].clone()
            qw, err, tail_num = quant_weights(w, args)

            each_layer_weights[c, :] = qw
            quant_error += err
            quant_num += len(qw.reshape(-1, 1))
            layer_tail_num += tail_num

        all_quant_num += quant_num
        all_quant_error += quant_error
        all_tail_num += layer_tail_num

        checkpoint[each_layer] = each_layer_weights
        print('layer quant RMSE: %.4e' % np.sqrt(quant_error / quant_num))
        print('layer tail region percentage: %.2f' % (layer_tail_num / quant_num * 100))
        
    rmse = np.sqrt(all_quant_error / all_quant_num)
    print('\ntotal quant RMSE: %.4e' % rmse)
    print('toatl tail region percentage: %.2f' % (all_tail_num / all_quant_num * 100))

    return checkpoint, rmse


def quant_weights(w, args):
    '''
    Quantize a tensor of weights 
    '''
    bkp_ratio = 0.0
    if args.wei_quant_scheme.startswith('uni'):
        # uniform symmetric quantization
        qw = uniform_symmetric_quantizer(w, bits=args.wei_bits)
    else: 
        try:
            pw_opt = int(args.wei_quant_scheme[-1])
        except:
            print('PWLQ options: \n  pw-1 with overlapping regions; pw-2 with non-overlapping regions')
            raise RuntimeError("Please specify an option for PWLQ, like 'pw-2' !!!")
        
        # piecewise linear quantization (PWLQ)
        qw, bkp_ratio, _ = piecewise_linear_quant(w, 
                                bits=args.wei_bits, scale_bits=args.scale_bits, 
                                break_point_approach=args.break_point, pw_opt=pw_opt, 
                                approximate=args.approximate)

    # bias correction
    if args.bias_corr:
        mean_w, std_w = torch.mean(w), torch.std(w)
        mean_diff = mean_w - torch.mean(qw)
        std_ratio = torch.div(std_w, torch.std(qw) + 1e-12)
        if args.scale_bits > 0:
            scale_levels = 2 ** args.scale_bits
            mean_diff = torch.round(torch.mul(mean_diff, scale_levels)) / scale_levels
            std_ratio = torch.round(torch.mul(std_ratio, scale_levels)) / scale_levels

        qw = torch.mul(qw + mean_diff, std_ratio)

    err = float(torch.sum(torch.mul(qw - w, qw - w)))

    abs_max = torch.max(torch.abs(w))
    break_point = abs_max * bkp_ratio
    tail_num = np.sum(torch.abs(w).detach().numpy() > float(break_point))

    return qw, err, tail_num