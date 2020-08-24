# This file is part of PWLQ repository.
# Copyright (c) Samsung Semiconductor, Inc.
# All rights reserved.

import torch
import numpy as np
from .uniform import *


##########################################################################################
####  Piecewise Linear Quantization (PWLQ)
##########################################################################################

# # piecewise linear quantization on range [-m, m] with breakpoint p: 
#       4 pieces: [-m, -p], [-p, 0], [0, p], [p, m]
#   option 1: overlapping 
#       [-p, p], [-m, m] symmetric with b bits signed, zero offsets
#   option 2: non-overlapping
#       [-p, p] symmetric with b bits signed
#       [-m, -p] and [p, m] asymmetric with (b - 1) bits unsigned, non-zero offset p  


def piecewise_linear_quant(w, bits=4.0, scale_bits=0.0, 
                        break_point_approach='norm', pw_opt=2, approximate=False):
    '''
    Piecewise Linear Quantization (PWLQ)
    '''
    if break_point_approach == 'search': 
        ## coarse-to-fine grid search for minimizing quantization error
        qw, bkp_ratio, err = pwlq_coarse2fine_grid_search(w, 
                                bits=bits, scale_bits=scale_bits, pw_opt=pw_opt)
    else:
        ## assumption: w is stastifying Gaussian or Laplacian distribution
        # break_point_approach = 'norm' (Gaussian) or 'laplace' (Laplacian)
        std_w = torch.std(w) + 1e-12
        abs_max = torch.max(torch.abs(w))
        abs_max_normalized = abs_max / std_w
        break_point_normalized = find_optimal_breakpoint(abs_max_normalized, 
                                    pw_opt=pw_opt, dist=break_point_approach, approximate=approximate)
        bkp_ratio = break_point_normalized / abs_max_normalized
        break_point = bkp_ratio * abs_max
        err, qw = pwlq_quant_error(w, bits, scale_bits, abs_max, break_point, pw_opt)
    return qw, bkp_ratio, err


def pwlq_quant_error(w, bits, scale_bits, abs_max, break_point, pw_opt):
    '''
    Piecewise linear quantization (PWLQ) with two options: overlapping or non-overlapping
    '''
    ## option 1: overlapping
    if pw_opt == 1:
        qw_tail = uniform_symmetric_quantizer(w, 
            bits=bits, scale_bits=scale_bits, minv=-abs_max, maxv=abs_max)
        qw_middle = uniform_symmetric_quantizer(w, 
            bits=bits, scale_bits=scale_bits, minv=-break_point, maxv=break_point)
        
        qw = torch.where(-break_point < w, qw_middle, qw_tail)
        qw = torch.where(break_point > w, qw, qw_tail)
    
    ## option 2: non-overlapping
    if pw_opt == 2:
        qw_tail_neg = uniform_affine_quantizer(w, 
            bits=bits-1, scale_bits=scale_bits, minv=-abs_max, maxv=-break_point)
        qw_tail_pos = uniform_affine_quantizer(w, 
            bits=bits-1, scale_bits=scale_bits, minv=break_point, maxv=abs_max)
        qw_middle = uniform_symmetric_quantizer(w, 
            bits=bits, scale_bits=scale_bits, minv=-break_point, maxv=break_point)
    
        qw = torch.where(-break_point < w, qw_middle, qw_tail_neg)
        qw = torch.where(break_point > w, qw, qw_tail_pos)

    err = torch.sqrt(torch.sum(torch.mul(qw - w, qw - w)))
    return err, qw


def pwlq_coarse2fine_grid_search(w, bits=4.0, scale_bits=0.0, search_range=10, pw_opt=2):
    '''
    Coarse-to-fine grid search for optimal breakpoint to minimize the quantization error
    Search for optimal breakpoint ratio: bkp_ratio = break_point / abs_max
    '''
    min_err = 10000
    abs_max = torch.max(torch.abs(w))
    
    ## first stage
    for bkp_ratio in np.arange(0.1, 1.0, 0.1):
        break_point = bkp_ratio * abs_max
        err, qw = pwlq_quant_error(w, bits, scale_bits, abs_max, break_point, pw_opt)
        if err < min_err:
            min_err = err
            best_ratio = bkp_ratio
            best_qw = qw

    ## second stage
    ratio_start, ratio_end = best_ratio - 0.01 * search_range, best_ratio + 0.01 * search_range
    for bkp_ratio in np.arange(ratio_start, ratio_end, 0.01):
        break_point = bkp_ratio * abs_max
        err, qw = pwlq_quant_error(w, bits, scale_bits, abs_max, break_point, pw_opt)
        if err < min_err:
            min_err = err
            best_ratio = bkp_ratio
            best_qw = qw

    ## third stage
    ratio_start, ratio_end = best_ratio - 0.001 * search_range, best_ratio + 0.001 * search_range
    for bkp_ratio in np.arange(ratio_start, ratio_end, 0.001):
        break_point = bkp_ratio * abs_max
        err, qw = pwlq_quant_error(w, bits, scale_bits, abs_max, break_point, pw_opt)
        if err < min_err:
            min_err = err
            best_ratio = bkp_ratio
            best_qw = qw
        
    return best_qw, best_ratio, min_err


##########################################################################################
####  Optimal breakpoint under Guassian or Laplacian assumption
##########################################################################################

def derivative_quant_err(m, p, dist='norm', pw_opt=2):  
    '''
    Compute the derivative of expected variance of quantization error
    '''
    from scipy.stats import norm, laplace
    if dist == 'norm':
        cdf_func = norm.cdf(p)
        pdf_func = norm.pdf(p)
    elif dist == 'laplace':  
        # https://en.wikipedia.org/wiki/Laplace_distribution
        cdf_func = laplace.cdf(p, 0, np.sqrt(0.5))   
        pdf_func = laplace.pdf(p, 0, np.sqrt(0.5)) # pdf(p, a, b) has variance 2*b^2
    else:
        raise RuntimeError("Not implemented for distribution: %s !!!" % dist) 
    
    ## option 1: overlapping
    if pw_opt == 1: 
        # quant_err = [F(p) - F(-p)] * p^2 + 2*[F(m) - F(p)] * m^2
        df_dp = 2 * pdf_func * (p * p - m * m) + 2 * p * (2 * cdf_func - 1.0)
    ## option 2: non-overlapping
    else:  
        # quant_err = [F(p) - F(-p)] * p^2 + 2*[F(m) - F(p)] * (m - p)^2
        df_dp = p - 2 * m + 2 * m * cdf_func + m * pdf_func * (2 * p - m) 

    return df_dp


def gradient_descent(m, pw_opt, dist, lr=0.1, max_iter=100, tol=1e-3):
    '''
    Gradient descent method to find the optimal breakpoint
    '''
    p = m / 2.0
    err, iter_num = 1, 0
    while err > tol and iter_num < max_iter:
        grad = derivative_quant_err(m, p, pw_opt=pw_opt, dist=dist)
        p = p - lr * grad
        err = np.abs(grad)
        iter_num += 1
    return p


def binary_search(m, pw_opt, dist, max_iter=100, tol=1e-3):
    '''
    Binary search method to find the optimal breakpoint
    '''
    left, right = 0, m
    fl = derivative_quant_err(m, left, pw_opt=pw_opt, dist=dist)
    fr = derivative_quant_err(m, right, pw_opt=pw_opt, dist=dist)
    err, iter_num = 1, 0
    while err > tol and iter_num < max_iter:
        mid = (right - left) / 2.0 + left
        fm = derivative_quant_err(m, mid, pw_opt=pw_opt, dist=dist)
        if fm * fl > 0:
            left = mid
            fl = fm
        else:
            right = mid
            fr = fm
        err = np.abs(fm)
        iter_num += 1
    return mid 


def find_optimal_breakpoint(m, pw_opt=2, dist='norm', approximate=True):
    '''
    Find the optimal breakpoint for PWLQ: approximated VS numerical solution
    '''
    m = float(m)
    ## linear approximated solution O(1)
    if approximate:
        assert(pw_opt == 2 and dist in ['norm', 'laplace'])
        if dist.startswith('norm') and pw_opt != 1:
            # Approximated version for Gaussian
            coef = 0.86143114  
            inte = 0.607901097496529 
            break_point = np.log(coef * m + inte)
        elif dist.startswith('laplace') and pw_opt != 1:
            # Approximated version for Laplacian
            coef = 0.80304483
            inte = -0.3166785508381478
            break_point = coef * np.sqrt(m) + inte
    ## numeric solution: gradient descent or binary search
    else:
        # here we use binary search: O(log m)
        break_point = binary_search(m, pw_opt, dist)
        # # or we can use gradient descent
        # break_point = gradient_descent(m, pw_opt, dist)

    ## add ramdon noise
    if 'noise' in dist:
        noise = abs(float(dist.split('-')[-1]))
        rand_num = np.random.uniform(-1.0, 1.0)
        random_noise = noise if rand_num >= 0.0 else -1.0 * noise
        break_point *= (1 + random_noise)
    
    return break_point