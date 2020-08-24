# This file is part of PWLQ repository.
# Copyright (c) Samsung Semiconductor, Inc.
# All rights reserved.

import torch

##########################################################################################
####  Fold Batch Normalization
##########################################################################################

# References: 
#   https://arxiv.org/abs/1712.05877 

# before folding:
#   x_bn_out = gamma * (x_bn_in - mean) / sqrt(var + eps) + beta
#   x_bn_in = conv(w, x_conv_in) with weights w and bias b = 0
# 
# after folding (remove bn):
#   x_bn_out = conv(w_new, x_conv_in) with new weights w_new and new bias b_new
#   w_new = gamma * w / sqrt(var + eps)
#   b_new = beta - gamma * mean / sqrt(var + eps)
# 
# after folding (remain bn):
#   x_bn_in = conv(w_new, x_conv_in) with new weights w_new = gamma * w / sqrt(var + eps) and bias b = 0
# new bn variables: 
#   beta_new = beta - gamma * mean / sqrt(var + eps)
#   gamma_new = 1, mean_new = 0, var_new = 1 - eps
# then: 
#   x_bn_out = gamma_new * (x_bn_in - mean_new) / sqrt(var_new + eps) + beta_new 


def fold_batch_norm(checkpoint, arch='resnet50'):
    print('folding BN laryers for %s ... ' % arch)
    weight_layers, bn_layer_counts = [], 0
    layers_list = list(checkpoint.keys())
    
    if arch == 'resnet50':
        var_eps  = 1e-5
        for layer in layers_list:
            if '.running_mean' in layer:
                bn_base = layer.replace('.running_mean', '')
                if 'downsample' in layer:
                    conv_layer_num = int(bn_base.split('.')[-1]) - 1
                    conv_layer = '.'.join(bn_base.split('.')[:-1] + [str(conv_layer_num), 'weight'])
                else:
                    conv_layer = bn_base.replace('bn', 'conv') + '.weight'
                weight_layers.append(conv_layer)
                fold_batch_norm_for_one_layer(checkpoint, conv_layer, bn_base, var_eps)
                bn_layer_counts += 1
        print('conv and batch normalization layers: ', bn_layer_counts)
        assert(bn_layer_counts == 53)
        weight_layers.append('fc.weight')
    else:
        raise RuntimeError("Please implement BatchNorm folding for %s !!!" % arch) 

    return checkpoint, weight_layers


def fold_batch_norm_for_one_layer(checkpoint, conv_layer, bn_base, var_eps=1e-5):
    conv_weights = checkpoint[conv_layer].clone()

    bn_gamma = checkpoint[bn_base + '.weight'].clone()
    bn_beta = checkpoint[bn_base + '.bias'].clone()
    bn_mean = checkpoint[bn_base + '.running_mean'].clone()
    bn_var = checkpoint[bn_base + '.running_var'].clone()

    # x_bn_in = conv(w_new, x_conv_in) with new weights w_new = gamma * w / sqrt(var + eps) and bias b=0
    for c in range(conv_weights.size()[0]):
        conv_weights[c, :, :, :] *= bn_gamma[c] * torch.rsqrt(torch.add(bn_var[c], var_eps))            
    checkpoint[conv_layer] = conv_weights
    
    # new bn variables: beta_new = beta - gamma * mean / sqrt(var + eps), gamma_new = 1, mean_new = 0, var_new = 1 - eps
    checkpoint[bn_base + '.bias'] = bn_beta - bn_gamma * bn_mean * torch.rsqrt(torch.add(bn_var, var_eps))
    checkpoint[bn_base + '.weight'] = bn_gamma * 0.0 + 1.0
    checkpoint[bn_base + '.running_mean'] = bn_mean * 0.0
    checkpoint[bn_base + '.running_var'] = bn_var * 0.0 + 1.0 - var_eps 

    return checkpoint