import numpy as np
import copy
from tqdm import tqdm
from scipy import optimize
import scipy.io as sio


def t1_3params(t, p):
    c, k, t1 = p
    return c * (1 - k * np.exp(-t / t1))

def rms_3params_fitting(p, *params):
    t, data = params 
    return np.mean((t1_3params(t, p) - data)**2)


class MOLLIT1map:
    def __init__(self) -> None:
        self.type = 'Gaussian'


    def save_config(self, tvec, frames):
        pass


    def mestimation_abs(self, tvec=None, frames=None, mask=None):
    
        # Assumption: frames shape [H, W, L], where L is the number of points. For MOLLI we have L = 11
        if frames is None:
            frames = self.frames

        H, W, L = frames.shape
        
        # initialize results
        sdmap = np.zeros((H, W))
        S = np.zeros((H, W))
        null_index = np.zeros((H, W)) # null index is the index of the first point that requires polarity reversal

        pmap = np.zeros((H, W, 3))
        
        if mask is None:
            mask = np.ones((H, W))
        
        # sort the frames according to the TI values
        if tvec is None:
            tvec = self.config.tvec
        tvec_index = np.argsort(tvec)
        tvec_sorted = tvec[tvec_index]
        frames_sorted = frames[:, :, tvec_index]

        # pixel-wise fitting
        for tx in tqdm(range(H)):
            for ty in range(W):
                if mask[tx, ty] == 1:
                    
                    dvec = np.squeeze(frames_sorted[tx, ty, :])
                    # mask out the air and region outside the mask
                    if (max(dvec) < 10) or (mask[tx, ty] == 0):
                        continue

                    # MSE fitting
                    [p_mse, nl, _] = self.polarity_recovery_fitting(tvec_sorted, dvec)
                    dvec[:nl] = -dvec[:nl]
                    pred_mse = t1_3params(tvec_sorted, p=p_mse)
                    resid_mse = pred_mse - dvec
                    SD = np.median(np.abs(resid_mse)) * 1.48 # why 1.48?(Sigma estimation)
                    null_index[tx, ty] = nl;
                    S[tx, ty] = SD;
    
                    # Reweighting
                    dweight = np.ones(dvec.shape)
                    if self.type == 'Gaussian':
                        pres = p_mse
                    else:
                        raise NotImplementedError
                    
                    pmap[tx, ty, :] = pres
                    sdmap[tx, ty] = self.MOLLIComputeSD(pres, tvec_sorted, SD)

        return pmap, sdmap, null_index, S    

    def polarity_recovery_fitting(self, t_vec, data_vec, p0=[150, 2, 1000]):
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
        assert len(t_vec) == len(data_vec), "TI vector and pixel-wise data vector should have the same length"
        minD_idx = np.argmin(data_vec) + 2
        inds = minD_idx if minD_idx < 8 else 8

        # iterarively revert the polarity
        fitting_err_vals = np.zeros(inds)
        fitting_results = np.zeros((inds, 3))

        for test_num in range(inds):
            data_vec_temp = copy.deepcopy(data_vec)
            data_vec_temp = np.concatenate((-data_vec_temp[:test_num], data_vec_temp[test_num:]))
            # data_vec_temp[:inds[test_num]] = - 1.0 * data_vec_temp[:inds[test_num]] # invert the polarity

            # use the grid search solver
            minimum = optimize.fmin(rms_3params_fitting, p0, args=(t_vec, data_vec_temp), maxfun=2000, disp=False, full_output=True)
            fitting_err_vals[test_num] = minimum[1] # value of function at the minimum, t1 error
            fitting_results[test_num, :] = minimum[0] # three parameters(A, B, T1*) that minimize the t1 error
        
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
        dc = 1 - k * np.exp(-tvec * (k - 1) / T1) # derivative of c
        dk = -c * np.exp(- tvec * (k - 1 ) / T1) + c * k / T1 * tvec * np.exp( - tvec * (k -1)/T1); #TODO: check the dot products
        dT1 = - c * k * np.exp( -tvec * (k -1) / T1) * tvec * (k-1) / T1 / T1;
        for j in range(len(tvec)):
            D[:,:,j] = np.dot(np.array([dT1[j], dc[j], dk[j]])[:, None], np.array([dT1[j], dc[j], dk[j]])[None, :]) 
        D = np.squeeze(np.sum(D, axis=-1)) / s / s
        Dinv = np.linalg.inv(D + eps * np.eye(3))
        SD = np.sqrt(Dinv[0, 0])
        return SD
