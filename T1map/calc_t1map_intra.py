# -*- coding: utf-8 -*-
"""
voxelwise cardiac T1 mapping

contributors: Yoon-Chul Kim, Khu Rai Kim, Hyelee Lee
"""
import logging
import hydra
import numpy as np
import copy

from scipy.optimize import curve_fit
from scipy import optimize


hydralog = logging.getLogger(__name__)

def func_orig(x, a, b, c):
    # return a*(1-np.exp(-b*x)) + c
    return a * (1 - np.exp(- (x / b))) + c
    # return a - b * np.exp(-x / c)

def t1_3params(t, p):
    c, k, t1 = p
    return c * (1 - np.exp(-t / t1)) + k
    # return c - k * np.exp(-t / t1)


def rms_3params_fitting(p, *params):
    t, data = params
    return np.mean((t1_3params(t, p) - data)**2)


def linear(x, a, b):
    return a * x + b


def map_fitting(j, ir_img, inversiontime, type=False):

    nx, ny, nti = ir_img.shape
    inversiontime = np.asarray(inversiontime)
    y = np.zeros(nti)
    r = int(j/ny)
    c = int(j%ny)

    for tino in range(nti):
        y[tino] = ir_img[r,c,tino]
    # y = np.sort(y)
    threshold = 2000
    
    yf = copy.deepcopy(y)
    sq_err = 100000000.0
    curve_fit_success = False
    try:
        if type == 'single_exp':
            p0_initial = [300, 1000, 50]
            popt,pcov = curve_fit(func_orig, inversiontime, yf, p0=p0_initial)
        elif type == 'linear':
            p0_initial = [(yf[1] - yf[0])/(inversiontime[1] - inversiontime[0]), yf[0]]
            popt,pcov = curve_fit(linear, inversiontime, yf, p0=p0_initial)
        elif type == 'dual_exp':
            raise NotImplementedError
        
    except RuntimeError:
        hydralog.debug(f"Curve fitting failed for {r}, {c}")
        popt = p0_initial

    if type == 'single_exp':
        a1, b1, c1 = popt
        yf_est = a1 * (1 - np.exp(- (inversiontime / b1))) + c1
    elif type == 'linear':
        a1, b1 = popt
        c1 = 0
        yf_est = a1 * inversiontime + b1

    sq_err_curr = np.sum((yf_est - yf)**2, dtype=np.float32)

    if sq_err_curr < sq_err:
        curve_fit_success = True
        sq_err = sq_err_curr
        a1_opt = a1
        b1_opt = b1
        c1_opt = c1
        sq_reside = np.median(np.abs(yf_est - yf))

    if not curve_fit_success:
        a1_opt = p0_initial[0]
        b1_opt = p0_initial[1]
        c1_opt = p0_initial[2]
        sq_reside = np.nan

    return a1_opt, b1_opt, c1_opt, sq_reside


def calculate_T1map(ir_img, inversiontime, type):
    
    nx, ny, nti = ir_img.shape


    fitted_map = np.zeros([nx, ny, 3])

    sqerrmap = np.zeros([nx, ny])


    for j in range(nx*ny):
        r = int(j / ny)
        c = int(j % ny)
        p1, p2, p3, err = map_fitting(j, ir_img, inversiontime, type)
        fitted_map[r, c, :] = [p1, p2, p3]
        sqerrmap[r, c] = err

    return fitted_map, sqerrmap