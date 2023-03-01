import os
import numpy as np
import pickle
import scipy.io as sio
from T1map import MOLLIT1map
from T1mapParallel import MOLLIT1mapParallel
import calc_t1map_py as t1_py

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
    t1map = MOLLIT1mapParallel()
    inversion_img, pmap, sdmap, null_index, S = t1map.mestimation_abs(tvec, frames)
    t1_params_pre = t1_py.calculate_T1map(frames, tvec)
    a = t1_params_pre[:, :, 0]
    b = t1_params_pre[:, :, 1]
    c = t1_params_pre[:, :, 2]
    t1 = (1 / b) * (a / (a + c) - 1)

    # re = {}
    # re['pmap'] = pmap
    # re['sdmap'] = sdmap
    # re['null_index'] = null_index
    # re['S'] = S
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(t1, handle)
    