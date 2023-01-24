import os
import numpy as np
import pickle
import scipy.io as sio
from T1map import MOLLIT1map
from T1mapParallel import MOLLIT1mapParallel

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
    pmap, sdmap, null_index, S = t1map.mestimation_abs(tvec, frames)
    re = {}
    re['pmap'] = pmap
    re['sdmap'] = sdmap
    re['null_index'] = null_index
    re['S'] = S
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(re, handle)
    