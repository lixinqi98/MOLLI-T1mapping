import numpy as np
import torch
import math
import itertools
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import torch.nn.functional as F
# from ..T1map.T1mapParallel import MOLLIT1mapParallel
from T1mapParallel import MOLLIT1mapParallel
from utils import *
import SimpleITK as sitk
class Optimize:
    def __init__(self, device='cpu') -> None:
        self.t1map = MOLLIT1mapParallel()
        self.device = device
        self.grad_u_kernel = gradient_kernel_3d(method="default", axis=0)
        self.grad_v_kernel = gradient_kernel_3d(method="default", axis=1)

    def save_data(self, I, S):
        self.I = I
        self.S = S

    def ini_synthesis(self, tvec, data):
        self.t1map.initial = True
        inversion_recovery_img, pmap, sdmap, null_index, S = self.t1map.mestimation_abs(tvec, data)
        return inversion_recovery_img, pmap, sdmap, null_index, S
    
    def correlation_coefficient(self, data, threshold=0.75):
        weight = np.zeros(data.shape[:-1])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x = data[i, j, :]
                neighbor = data[max(0, i-1):min(data.shape[0], i+2), max(0, j-1):min(data.shape[1], j+2), :]
                corr = 0
                for idx1, idx2 in itertools.product([0,1],[0,1]):
                    corr += np.corrcoef(x, neighbor.reshape(-1, data.shape[-1]))[0, 1]
                weight[i,j] = corr / 4.0
        weight[weight < threshold] = 0
        weight = np.nan_to_num(weight)
        return weight

        

    def synthesis_gradient(self, data, sigma=1, order=2):
        gradient = gaussian_filter(data, sigma=sigma, order=order)
        return gradient
    
    def derivative(self, M):
        dx = ndimage.convolve(M, self.grad_u_kernel, mode='constant', cval=0.0)
        dy = ndimage.convolve(M, self.grad_v_kernel, mode='constant', cval=0.0)
        return dx ** 2 + dy ** 2

    def energy_derivative(self, I, M, S, alpha=1, beta=1):
        w = self.correlation_coefficient(I)
        np.nan_to_num(w)
        second_order_derivative = self.synthesis_gradient(M)
        w_t = np.repeat(w[..., np.newaxis], I.shape[-1], axis=-1)
        tmp = alpha * w_t * second_order_derivative - (1 + beta) * M + I + beta * S
        return tmp

    
    def energy(self, M, I, S, alpha=1, beta=1):

        term1 = (I - M) ** 2
        w = alpha * self.correlation_coefficient(I)
        np.nan_to_num(w)
        w_t = np.repeat(w[..., np.newaxis], I.shape[-1], axis=-1)
        term2 = w_t * self.derivative(M)
        term3 = beta * (M - np.abs(S)) ** 2
        return np.sum(term1 + term2 + I + term3)


    def view(self, img):
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
        image_viewer = sitk.ImageViewer()
        image_viewer.SetTitle('grid using ImageViewer class')

        image_viewer.SetApplication('/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP')
        # image_viewer.SetApplication('/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx')
        image_viewer.Execute(img)
