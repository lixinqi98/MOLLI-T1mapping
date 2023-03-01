import os
import numpy as np
import pickle
import copy
import warnings
import scipy.io as sio
import matplotlib.pyplot as plt
from Optimize import Optimize
from scipy import optimize
from calc_t1map_py import *

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def load_orig_file(mat_fname):
    mat_contents = sio.loadmat(mat_fname)
    tvec = np.squeeze(mat_contents['tvec']).astype(np.float32)
    frames = np.squeeze(mat_contents['img']).astype(np.float32)
    return tvec, frames


__file__ = '/Users/mona/Documents/repo/voxelmorph-test/data/LisbonD8_postconT1w_mat/POSTCON1_T1MAP_SAX5_0123_T1w.mat'
if __name__ == "__main__":
    # load the data
    # filename = os.path.join(os.path.dirname(__file__), "data", "MOLLI.mat")
    filename = __file__
    tvec, frames = load_orig_file(filename)
    min_ti = np.argmin(tvec)
    max_ti = np.argmax(tvec)
    initial_frame = np.dstack((frames[:, :, min_ti], frames[:, :, max_ti]))
    t1_params = calculate_T1map(initial_frame, np.array([tvec[min_ti], tvec[max_ti]]))

    opt = Optimize()
    inversion_recovery_img, pmap, sdmap, null_index, S = opt.ini_synthesis(tvec, frames)
    inversion_recovery_img = frames
    # initial_T1 = S
    
    I = frames
    S = inversion_recovery_img
    initial_M = copy.copy(inversion_recovery_img)
    re = opt.energy_derivative(initial_M, I, S)
    energy = opt.energy(initial_M, I, S)
    result = optimize.fmin(opt.energy, initial_M, args=(
                    I, S), maxfun=1000, disp=True, full_output=True)
    print(result)
    