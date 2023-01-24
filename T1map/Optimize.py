import numpy as np
import torch
import math
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
# from ..T1map.T1mapParallel import MOLLIT1mapParallel
from T1mapParallel import MOLLIT1mapParallel

class Optimize:
    def __init__(self, device='cpu') -> None:
        self.t1map = MOLLIT1mapParallel()
        self.device = device


    def synthesis(self, tvec, data):
        initial_data = np.dstack((data[:,:,0], data[:,:,-1]))
        inversion_recovery_img, pmap, sdmap, null_index, S = self.t1map.mestimation_abs(tvec, data)
        return inversion_recovery_img, pmap, sdmap, null_index, S
    
    def correlation_coefficient(self, data, window_size=2, threshold=0.75):
        H, W, D = data.shape
        Ii = torch.from_numpy(data[..., None]).to(self.device).permute(2, 3, 0, 1)
        Ji = torch.from_numpy(data[..., None]).to(self.device).permute(2, 3, 0, 1)
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [window_size] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        
        cc[cc < threshold] = 0
        tmp = torch.squeeze(cc).permute(1, 2, 0).cpu().numpy()
        return tmp[:H, :W, :D]

    def synthesis_gradient(self, data, sigma=1, order=2):
        gradient = gaussian_filter(data, sigma=sigma, order=order)
        return gradient

    def energy_derivative(self, I, M, S, alpha=1, beta=1):
        weight = self.correlation_coefficient(I)
        second_order_derivative = self.synthesis_gradient(M)
        tmp = alpha * weight * second_order_derivative - (1 + beta) * M + I + beta * S

    
    def energy(self, I, M, S, alpha=1, beta=1):
        term1 = (I - M) ** 2
        term2 = alpha * self.correlation_coefficient(I)
        term2 = term2 * (self.synthesis_gradient(M, order=1) ** 2)
        term3 = beta * (M - S) ** 2
        return term1 + term2 + term3

