import os
import numpy as np
import pickle
import scipy.io as sio
from Optimize import Optimize

def load_orig_file(mat_fname):
    mat_contents = sio.loadmat(mat_fname)
    tvec = np.squeeze(mat_contents['tvec_post']).astype(np.float32)
    frames = np.squeeze(mat_contents['volume_post']).astype(np.float32)
    return tvec, frames

__file__ = '/Users/mona/Documents/data/registration/voxelmorph/MOLLI_original/0379217_20131018_MOLLI.mat'
if __name__ == "__main__":
    # load the data
    # filename = os.path.join(os.path.dirname(__file__), "data", "MOLLI.mat")
    filename = __file__
    tvec, frames = load_orig_file(filename)
    print(tvec.shape)
    
    