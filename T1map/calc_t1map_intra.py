# -*- coding: utf-8 -*-
"""
voxelwise cardiac T1 mapping

contributors: Yoon-Chul Kim, Khu Rai Kim, Hyelee Lee
"""

import numpy as np
import copy

from scipy.optimize import curve_fit
from scipy import optimize

# Y0 = 0
# Plateau = 2000

def func_orig(x, a, b, c):
    # return a*(1-np.exp(-b*x)) + c
    return a * (1 - np.exp(- (x / b))) + c
    # return a - b * np.exp(-x / c)

# def dual_fitting(input, PercentFast, KFast, KSlow):
    
#     X = input[0:-2]
#     # print(input, X, Y0, Plateau)
#     SpanFast=(Plateau - Y0)*PercentFast*.01
#     SpanSlow=(Plateau - Y0)*(100-PercentFast)*.01
#     # X = np.sort(x - np.min(x))
#     # print(Y0, Plateau, X, SpanFast, SpanSlow)
#     # Y= Plateau - SpanFast*np.exp(-KFast*X) - SpanSlow*np.exp(-KSlow*X)
#     return Plateau - SpanFast*np.exp(-KFast*X) - SpanSlow*np.exp(-KSlow*X)

def dual_fitting_help(p, *params):
    X, yf = params
    Plateau = np.max(yf)
    Y0 = np.min(yf)
    PercentFast, KFast, KSlow = p
    # print(input, X, Y0, Plateau)
    SpanFast=(Plateau - Y0)*PercentFast*.01
    SpanSlow=(Plateau - Y0)*(100-PercentFast)*.01
    # X = np.sort(x - np.min(x))
    # print(Y0, Plateau, X, SpanFast, SpanSlow)
    # Y= Plateau - SpanFast*np.exp(-KFast*X) - SpanSlow*np.exp(-KSlow*X)
    yf_est = Plateau - SpanFast*np.exp(-KFast*X) - SpanSlow*np.exp(-KSlow*X)
    return np.mean((yf_est - yf)**2)

def t1_3params(t, p):
    c, k, t1 = p
    return c * (1 - np.exp(-t / t1)) + k
    # return c - k * np.exp(-t / t1)


def rms_3params_fitting(p, *params):
    t, data = params
    return np.mean((t1_3params(t, p) - data)**2)


def calc_t1value(j, ir_img, inversiontime, dual=False):

    nx, ny, nti = ir_img.shape
    inversiontime = np.asarray(inversiontime)
    y = np.zeros(nti)
    r = int(j/ny)
    c = int(j%ny)
    if not dual:
        p0_initial = [300, 1000, 50]
    else:
        # initial guess for dual exponential model
        p0_initial = [80, 0.001, 0.05]
    for tino in range(nti):
        y[tino] = ir_img[r,c,tino]
    # y = np.sort(y)
    threshold = 2000
    
    yf = copy.deepcopy(np.sort(y))
    sq_err = 100000000.0
    curve_fit_success = False
    try:
        if not dual:
            popt,pcov = curve_fit(func_orig, inversiontime, yf, p0=p0_initial)
        else:
            # print(f"Use the dual fitting model, {np.append(inversiontime, [np.min(yf), np.max(yf)])}")

            # popt,pcov = curve_fit(dual_fitting, np.append(inversiontime, [np.min(yf), np.max(yf)]), yf, p0=p0_initial)
        # print("Fitting success")
        # popt,pcov = curve_fit(func_orig, inversiontime, yf)
        # minimum = optimize.fmin(dual_fitting, x0=p0_initial, args=(
        #     inversiontime, yf), maxfun=2000, disp=False, full_output=True)
            minimum = optimize.fmin(dual_fitting_help, x0=p0_initial, args=(
                inversiontime, yf), maxfun=2000, disp=True, full_output=True)
            popt = minimum[0]
    except RuntimeError:
        # print("Error - curve_fit failed")
        # curve_fit_success = False
        popt = p0_initial


    a1 = popt[0]
    b1 = popt[1]
    c1 = popt[2]
    if not dual:
        yf_est = a1 * (1 - np.exp(- (inversiontime / b1))) + c1
    else:
        Plateau = np.max(yf)
        Y0 = np.min(yf)
        SpanFast=(Plateau - Y0)*a1*.01
        SpanSlow=(Plateau - Y0)*(100-a1)*.01
        yf_est = a1 - SpanFast*np.exp(-b1*inversiontime) - SpanSlow*np.exp(-c1*inversiontime)
    sq_err_curr = np.sum((yf_est - yf)**2, dtype=np.float32)

    if sq_err_curr < sq_err:
        curve_fit_success = True
        sq_err = sq_err_curr
        a1_opt = a1
        b1_opt = b1
        c1_opt = c1
        sq_reside = np.median(np.abs(yf_est - yf))

    if not curve_fit_success:
        a1_opt = 0
        b1_opt = np.finfo(np.float32).max
        c1_opt = 0
        sq_reside = 0

    return a1_opt, b1_opt, c1_opt, sq_reside


def calculate_T1map(ir_img, inversiontime, dual=False):
    
    nx, ny, nti = ir_img.shape
    t1map = np.zeros([nx, ny, 3])
    sqerrmap = np.zeros([nx, ny])
    ir_img = copy.copy(ir_img)
    if inversiontime[-1] == 0:
        inversiontime = inversiontime[0:-1]
        nTI = inversiontime.shape[0]
        if nti > nTI:
            ir_img = ir_img[:,:,0:nTI]

    for j in range(nx*ny):
        r = int(j / ny)
        c = int(j % ny)
        if r == nx //2 and c == ny //2:
            print("stop")
        p1, p2, p3, err = calc_t1value(j, ir_img, inversiontime, dual)
        t1map[r, c, :] = [p1, p2, p3]
        sqerrmap[r, c] = err

    return t1map, sqerrmap


def getNLSStruct_v3(*args, **kwargs):
    """
    nlsS = getNLSStruct( extra, dispOn, zoom)
    nlsS : (datatype) dictionary

    extra['key']

    dispOn: 1 - display the struct at the end
            0 (or omitted) - no display
    zoom  : 1 (or omitted) - do a non-zoomed search
            x>1 - do an iterative zoomed search (x-1) times (slower search)
            NOTE: When zooming in, convergence is not guaranteed.

    Data Model    : a + b*exp(-TI/T1)

    written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
    (c) Board of Trustees, Leland Stanford Junior University
    """

    extra = args[0]
    if len(args) > 1:
        dispOn = args[1]
    elif kwargs.get('dispOn'):
        dispOn = kwargs['dispOn']
    if len(args) > 2:
        zoom = args[1]
    if kwargs.get('zoom'):
        zoom = kwargs[1]

    nlsS = dict()
    nlsS['tVec'] = extra['tVec']
    nlsS['N'] = len(nlsS['tVec'])
    nlsS['T1Vec'] = extra['T1Vec']
    nlsS['T1Start'] = nlsS['T1Vec'][0]
    nlsS['T1Stop'] = nlsS['T1Vec'][-1]
    nlsS['T1Len'] = len(nlsS['T1Vec'])

    # Set the number of times you zoom the grid search in, 1 = no zoom
    # Setting this greater than 1 will reduce the step size in the grid search
    # (and slow down the fit significantly)

    nargin = len(args) + len(kwargs)
    if nargin < 3:
        nlsS['nbrOfZoom'] = 2
    else:
        nlsS['nbrOfZoom'] = zoom

    if nlsS['nbrOfZoom'] > 1:
        nlsS['T1LenZ'] = 11  # Length of the zoomed search, default = 21

    # Set the help variables that can be precomputed:
    # alpha is 1/T1,
    # theExp is a matrix of exp(-TI/T1) for different TI and T1,
    # rhoNormVec is a vector containing the norm-squared of rho over TI,
    # where rho = exp(-TI/T1), for different T1's.

    alphaVec = 1. / nlsS['T1Vec']
    tVec_t = nlsS['tVec'].reshape(nlsS['tVec'].shape[0], -1)  # transpose
    tVec_t = tVec_t.astype('float64')
    nlsS['theExp'] = np.exp((-1) * tVec_t * alphaVec)  # datatype changed to matrix
    nlsS['rhoNormVec'] = np.conjugate(sum(np.power(nlsS['theExp'], 2), 0)) - 1 / nlsS['N'] * np.power(np.conjugate(sum(nlsS['theExp'], 0)), 2)

    if dispOn:
        # Display the structure for inspection
        print(nlsS)
    return nlsS


def rdNlsPr_v2(data, nlsS):
    """
        [T1Est, bMagEst, aMagEst, res] = rdNlsPr(data, nlsS)
        Finds estimates of T1, |a|, and |b| using a nonlinear least
        squares approach together with polarity restoration.
        The model +-|ra + rb*exp(-t/T1)| is used.
        The residual is the rms error between the data and the fit.

        INPUT:
        data - the absolute data to estimate from
        nlsS - struct containing the NLS search parameters and
                the data model to use

        written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
        (c) Board of Trustees, Leland Stanford Junior University

    """

    if len(data) != nlsS['N']:
        print(len(data))
        print(nlsS['N'])
        raise ValueError('nlsS.N and data must be of equal length!')

    # Make sure the data come in increasing TI-order
    tVec = sorted(nlsS['tVec'])
    order = nlsS['tVec'].argsort()

    data = np.squeeze(data)
    data = data[order]

    # Initialize variables
    modified_rdnls = True  # if set to False, it's the same as Barral's matlab code. If True, it's modified and improved version.

    if modified_rdnls:
        ncandidate = 3
    else:
        ncandidate = 2

    aEstTmp = np.zeros(ncandidate)
    bEstTmp = np.zeros(ncandidate)
    T1EstTmp = np.zeros(ncandidate)
    resTmp = 10000000.0 * np.ones(ncandidate)

    # Make sure data vector is a column vector
    # data = data.reshape(-1, 1)

    # Find the min of the data
    # minVal = np.min(data)
    minInd = np.argmin(data)

    # Fit
    try:
        nbrOfZoom = nlsS['nbrOfZoom']
    except:
        nbrOfZoom = 1  # No zoom

    '''     
    modified version of Barral 
    Barral method sometimes produces inaccurate fitting. 
    To overcome mis-fitting, we consider 3 cases rather than 2 cases. 
    '''
    for ii in range(ncandidate):
        theExp = nlsS['theExp'][order, :]

        if modified_rdnls:
            if ii == 0:
                # First, we set all elements up to and including
                # the smallest element to minus
                dataTmp = np.multiply(data, np.append((-1) * np.ones((minInd + 1, 1)), np.ones((nlsS['N'] - (minInd + 1), 1))))

            elif ii == 1:
                # Second, we set all elements up to (not including)
                # the smallest element to minus
                dataTmp = np.multiply(data, np.append((-1) * np.ones((minInd, 1)), np.ones((nlsS['N'] - minInd, 1))))

            elif ii == 2 and minInd >=2:  # this is added portion.
                dataTmp = np.multiply(data, np.append((-1)*np.ones((minInd-1, 1)), np.ones((nlsS['N']-(minInd-1), 1))))

            else:
                continue

        else:
            if ii == 0:
                # First, we set all elements up to and including
                # the smallest element to minus
                dataTmp = np.multiply(data, np.append((-1) * np.ones((minInd + 1, 1)),
                                                      np.ones((nlsS['N'] - (minInd + 1), 1))))

            elif ii == 1:
                # Second, we set all elements up to (not including)
                # the smallest element to minus
                dataTmp = np.multiply(data, np.append((-1) * np.ones((minInd, 1)),
                                                      np.ones((nlsS['N'] - minInd, 1))))

        # The sum of the data
        ySum = sum(dataTmp)

        # Compute the vector of rho'*t for different rho,
        # where rho = exp(-TI/T1) and y = dataTmp
        rhoTyVec = np.matmul(dataTmp, theExp) - 1 / nlsS['N'] * np.conjugate(np.sum(theExp, axis=0)) * ySum

        # rhoNormVec is a vector containing the norm-squared of rho over TI,
        # where rho = exp(-TI/T1), for different T1's.
        rhoNormVec = nlsS['rhoNormVec']

        # Find the max of the maximizing criterion
        cal = np.divide(np.power(abs(rhoTyVec), 2), rhoNormVec)
        # tmp = max(cal)
        ind = np.argmax(cal)

        T1Vec = nlsS['T1Vec']  # Initialize the variable

        if nbrOfZoom > 1 and False:  # Do zoomed search
            try:
                T1LenZ = nlsS['T1LenZ']  # For the zoomed search
            except:
                T1LenZ = 21  # For the zoomed search

            for _ in range(1, nbrOfZoom):
                if ind > 0 and ind < len(T1Vec) - 1:
                    T1Vec = np.conjugate(np.linspace(T1Vec[ind - 1], T1Vec[ind + 1], T1LenZ))
                elif ind == 0:
                    T1Vec = np.conjugate(np.linspace(T1Vec[ind], T1Vec[ind + 2], T1LenZ))
                else:
                    T1Vec = np.conjugate(np.linspace(T1Vec[ind - 2], T1Vec[ind], T1LenZ))

                # Update the variables
                alphaVec = 1 / T1Vec
                tVec = np.array(tVec)
                theExp = np.exp((-1) * np.matmul(tVec.reshape(-1, 1), np.conjugate(alphaVec).reshape(1, -1)))
                yExpSum = np.squeeze(np.matmul(dataTmp.reshape(1, -1), theExp))
                rhoNormVec = np.conjugate(np.sum(np.power(theExp, 2), axis=0)) - 1 / nlsS['N'] * np.power(np.conjugate(np.sum(theExp, axis=0)), 2)
                rhoTyVec = yExpSum - 1 / nlsS['N'] * np.conjugate(np.sum(theExp, axis=0)) * ySum

                # Find the max of the maximizing criterion
                cal = np.divide(np.power(abs(rhoTyVec), 2), rhoNormVec)
                # tmp = np.max(cal)
                ind = np.argmax(cal)
        else:
            tVec = np.array(tVec)

        # The estimated parameters
        T1EstTmp[ii] = T1Vec[ind]
        bEstTmp[ii] = rhoTyVec[ind] / rhoNormVec[ind]
        aEstTmp[ii] = 1 / nlsS['N'] * (ySum - bEstTmp[ii] * np.sum(theExp[:, ind]))

        # Compute the residual
        modelValue = aEstTmp[ii] + bEstTmp[ii] * np.exp((-1) * tVec / T1EstTmp[ii])
        # resTmp[ii] = 1 / sqrt(nlsS['N']) * np.linalg.norm(1 - np.divide(modelValue, dataTmp))
        # print(dataTmp.shape)
        # print(modelValue.shape)
        resTmp[ii] = np.linalg.norm(dataTmp - modelValue) # yck modified.

    # Finally, we choose the point of sign shift as the point giving
    # the best fit to the data, i.e. the one with the smallest residual
    # res = np.min(resTmp)
    ind = np.argmin(resTmp)
    aEst = aEstTmp[ind]
    bEst = bEstTmp[ind]
    T1Est = T1EstTmp[ind]

    a = -bEst
    b = 1./T1Est
    c = aEst + bEst

    return a, b, c


def calculate_T1map_pyrd(ir_img, inversiontime):
    '''
    implementation of Barral's method
    '''

    if inversiontime[-1] == 0:
        inversiontime = inversiontime[0:-1]

    if ir_img.shape[2] > inversiontime.shape[0]:
        ir_img = ir_img[:,:,0:ir_img.shape[2]-1]

    extra = {}
    extra['tVec'] = inversiontime
    extra['T1Vec'] = np.arange(1, 5001, 1)
    extra['N'] = inversiontime.shape[0]

    nlsS = getNLSStruct_v3(extra, 0)

    data = ir_img
    nbrow, nbcol, _ = data.shape
    nbslice = 1

    dshape = data.shape
    data = data.reshape(dshape[0], dshape[1], 1, -1)  # Make data a 4-D array regardless of number of slices
    mask = np.zeros((nbrow, nbcol, nbslice))
    tmpData = []
    nVoxAll = nbrow * nbcol
    maskInds_t = np.where(mask >= 0)
    for ii in range(data.shape[3]):
        tmpVol_t = data[:, :, :, ii]
        tmpData.append(np.array(tmpVol_t[maskInds_t]))

    tmpData = np.array(tmpData)
    tmpData = tmpData.transpose()
    data = tmpData

    nFitParams = 3
    ll_T1 = np.zeros((nVoxAll, nFitParams))
    for jj in range(nVoxAll):
        ll_T1[jj, :] = rdNlsPr_v2(data[jj, :], nlsS)

    # dims = [*mask.shape, nFitParams]
    im = np.zeros(mask.shape)

    T1 = np.zeros((*mask.shape, nFitParams))
    for ii in range(nFitParams):
        im[maskInds_t] = ll_T1[:, ii]
        T1[:, :, :, ii] = im

    t1map = np.squeeze(T1)

    return t1map

