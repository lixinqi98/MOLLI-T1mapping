import copy
import itertools
import logging
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import optimize

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


hydralog = logging.getLogger(__name__)


def t1_3params(t, p):
    c, k, t1 = p
    return c * (1 - k * np.exp(-t / t1))
    # return c - k * np.exp(-t / t1)


def rms_3params_fitting(p, *params):
    t, data = params
    return np.mean((t1_3params(t, p) - data)**2)


class MOLLIT1mapParallel:
    def __init__(self) -> None:
        self.type = 'Gaussian'
        self.initial = False


    def helper(self, args):
        tx, ty = args
        dvec_orig = np.squeeze(self.frames_sorted[tx, ty, :])

        # mask out the air and region outside the mask
        if (max(dvec_orig) <= 20) or (self.mask[tx, ty] == 0):
            return None

        # mask out the air and region outside the mask

        # MSE fitting
        # print(f"MSE fitting: (tx, ty) = ({tx}, {ty}), continue")
        dvec = copy.deepcopy(dvec_orig)
        if self.initial:
            ini_tvec = np.array([self.tvec_sorted[0], self.tvec_sorted[-2], self.tvec_sorted[-1]])
            ini_dvec = np.array([dvec[self.min_ti], dvec[self.tvec_idx[-2]], dvec[self.max_ti]])
            p_mse, nl, _ = self.polarity_recovery_fitting(ini_tvec, ini_dvec)
            assert nl == 1, "nl should be 1"
        else:
            p_mse, nl, _ = self.polarity_recovery_fitting(self.tvec_sorted, dvec)
        dvec[:nl] = -dvec[:nl]
        # p_mse_1 = copy.deepcopy(p_mse)
        # p_mse_1[-1] = p_mse[-1]*(p_mse[1] - 1)
        inv_recovery_img = t1_3params(self.tvec, p=p_mse)
        
        resid_mse = inv_recovery_img - dvec
        SD = np.median(np.abs(resid_mse)) * 1.48  # why 1.48?(Sigma estimation)

        # Reweighting
        dweight = np.ones(dvec.shape)
        if self.type == 'Gaussian':
            pres = p_mse
        else:
            raise NotImplementedError
        # print(f"MSE fitting: (tx, ty) = ({tx}, {ty}), SD = {SD}, p = {pres}")
        return inv_recovery_img, pres, SD, nl, tx, ty

    def mestimation_abs(self, tvec=None, frames=None, mask=None):

        # Assumption: frames shape [H, W, L], where L is the number of points. For MOLLI we have L = 11
        if frames is None:
            frames = self.frames

        H, W, L = frames.shape

        # initialize results
        sdmap = np.zeros((H, W))
        self.S = np.zeros((H, W))
        # null index is the index of the first point that requires polarity reversal
        self.null_index = np.ones((H, W))

        pmap = np.zeros((H, W, 3))

        if mask is None:
            self.mask = np.ones((H, W))

        # sort the frames according to the TI values
        if tvec is None:
            tvec = self.config.tvec
        tvec_index = np.argsort(tvec)
        self.tvec_idx = tvec_index
        self.tvec = tvec
        self.tvec_sorted = tvec[tvec_index]
        self.frames_sorted = frames[:, :, tvec_index]
        self.inversion_recovery = np.zeros((H, W, L))

        self.min_ti = tvec_index[0]
        self.max_ti = tvec_index[-1]
        # pixel-wise fitting
        items = itertools.product(range(H), range(W))
        processed_list = Parallel(n_jobs=-1)(delayed(self.helper)(i) for i in items)
        # processed_list = Parallel(n_jobs=int(num_cores/4), verbose=10)(delayed(self.helper)(i) for i in items)

        for result in processed_list:
            if result is not None:
                inv_recovery_img, pres, SD, nl, tx, ty = result
                self.null_index[tx, ty] = nl
                self.S[tx, ty] = SD
                self.inversion_recovery[tx, ty, :] = inv_recovery_img
                pmap[tx, ty, :] = pres
                sdmap[tx, ty] = self.MOLLIComputeSD(pres, self.tvec_sorted, SD)

        return self.inversion_recovery, pmap, sdmap, self.null_index, self.S

    def polarity_recovery_fitting(self, t_vec, data_vec, p0=[300, 2, 2000]):
        """Fits a 3-parameter T1 model s(t) = c * ( 1- k * exp(-t/T1) ) by grid search.
        The polarity of the first n observation data points are iteratively reverted to find the best null point.

        Args:
            t_vec (_type_): TI time vector
            data_vec (_type_): pixel-wise data vector
            p0 (_type_): initial value

        Returns:
            p: fitting parameters
            null_index: The data points before the null index require polarity reversal
        """
        assert len(t_vec) == len(
            data_vec), "TI vector and pixel-wise data vector should have the same length"


        # iterarively revert the polarity

        if self.initial:
            test_num = 1
            fitting_err_vals = np.zeros(test_num)
            fitting_results = np.zeros((test_num, 3))
            data_vec_flip = copy.copy(data_vec)
            data_vec_flip[0] = - 1.0 * data_vec_flip[0] # invert the polarity

            # use the grid search solver
            minimum = optimize.fmin(rms_3params_fitting, p0, args=(
                t_vec, data_vec_flip), maxfun=2000, disp=False, full_output=True)

            return minimum[0], 1, minimum[1]
        
        minD_idx = np.argmin(data_vec) + 2
        inds = minD_idx if minD_idx < 6 else 6
        inds = min(inds, len(t_vec))
        fitting_err_vals = np.zeros(inds)
        fitting_results = np.zeros((inds, 3))
        for test_num in range(inds):
            data_vec_temp = copy.copy(data_vec)
            data_vec_temp[test_num] = - 1.0 * data_vec_temp[test_num] # invert the polarity

            # use the grid search solver
            minimum = optimize.fmin(rms_3params_fitting, p0, args=(
                t_vec, data_vec_temp), maxfun=2000, disp=False, full_output=True)
            # value of function at the minimum, t1 error
            fitting_err_vals[test_num] = minimum[1]
            # three parameters(A, B, T1*) that minimize the t1 error
            fitting_results[test_num, :] = minimum[0]

        # find the best null point
        sorted_err_index = np.argsort(fitting_err_vals)
        for j in range(inds):
            p = fitting_results[sorted_err_index[j]]
            if p[1] < 1:
                null_index = 3
                continue
            else:
                null_index = sorted_err_index[j]
                break
        return p, null_index, fitting_err_vals[sorted_err_index]

    def MOLLIComputeSD(self, p, tvec, s, eps=1e-5):

        c, k, _ = p
        T1 = p[2] * (p[1] - 1)
        D = np.zeros((3, 3, len(tvec)))
        dc = 1 - k * np.exp(-tvec * (k - 1) / T1)  # derivative of c
        dk = -c * np.exp(- tvec * (k - 1) / T1) + c * k / T1 * tvec * \
            np.exp(- tvec * (k - 1)/T1)  # TODO: check the dot products
        dT1 = - c * k * np.exp(-tvec * (k - 1) / T1) * tvec * (k-1) / T1 / T1
        for j in range(len(tvec)):
            D[:, :, j] = np.dot(np.array([dT1[j], dc[j], dk[j]])[
                                :, None], np.array([dT1[j], dc[j], dk[j]])[None, :])
        D = np.squeeze(np.sum(D, axis=-1)) / s / s
        Dinv = np.linalg.inv(D + eps * np.eye(3))
        SD = np.sqrt(Dinv[0, 0])
        return SD
