import matplotlib.pyplot as plt
import argparse
import copy
import glob
import os
import shutil
import warnings
from pathlib import Path

import ants
import calc_t1map_inter as t1_inter
import calc_t1map_intra as t1_intra
import numpy as np
import pandas as pd
import pydicom
import scipy.io as sio
import SimpleITK as sitk
from Optimize import Optimize

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def load_orig_file(mat_fname):
    mat_contents = sio.loadmat(mat_fname)
    tvec = np.squeeze(mat_contents['tvec']).astype(np.float32)
    frames = np.squeeze(mat_contents['img']).astype(np.float32)
    return tvec, frames


def view(img):
    img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    image_viewer = sitk.ImageViewer()
    image_viewer.SetTitle('grid using ImageViewer class')

    image_viewer.SetApplication(
        '/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP')
    # image_viewer.SetApplication('/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx')
    image_viewer.Execute(img)


def t1fitting_inter(ini_frames, ini_tvec, tvec):
    t1_params_pre, sqerrormap = t1_inter.calculate_T1map(ini_frames, ini_tvec)

    a = t1_params_pre[:, :, 0]
    b = t1_params_pre[:, :, 1]
    c = t1_params_pre[:, :, 2]
    t1 = (1 / b) * (a / (a + c) - 1)

    inversion_recovery_img = np.zeros(
        (ini_frames.shape[0], ini_frames.shape[1], tvec.shape[-1]))
    for i in range(tvec.shape[-1]):
        x = tvec[i]
        inversion_recovery_img[..., i] = a*(1-np.exp(-b*x)) + c
    return inversion_recovery_img, t1, sqerrormap


def t1fitting_intra(ini_frames, ini_tvec, tvec):
    t1_params_pre, sqerrormap = t1_intra.calculate_T1map(ini_frames, ini_tvec)

    a = t1_params_pre[:, :, 0]
    b = t1_params_pre[:, :, 1]
    c = t1_params_pre[:, :, 2]
    t1 = (1 / b) * (a / (a + c) - 1)
    # t1 = a * (1 - np.exp(-x / b)) + c

    inversion_recovery_img = np.zeros(
        (ini_frames.shape[0], ini_frames.shape[1], tvec.shape[-1]))
    for i in range(tvec.shape[-1]):
        x = tvec[i]
        inversion_recovery_img[..., i] = a * (1 - np.exp(-(x / b))) + c
    return inversion_recovery_img, t1, sqerrormap


def synthetic(I, S, M, step_size=-0.1, iter=3, alpha=1, beta=1):
    opt = Optimize()
    energy = opt.energy(M, I, S) / np.size(M)
    print(f"Initial energy = {energy:.3f}")
    for itr in range(iter):
        step = opt.energy_derivative(M, I, S, alpha=alpha, beta=beta)
        new_M = M + step_size * step
        new_energy = opt.energy(new_M, I, S) / np.size(M)
        print(
            f"Iteration {itr}: energy = {new_energy:.3f}, diff = {abs(new_energy - energy):.3f}, step = {np.sum(step):.3f}")
        if new_energy > energy:
            break
        M = new_M
        energy = new_energy
    return M


def register(fixed, moving):
    registered = np.zeros_like(fixed)
    for slice in range(fixed.shape[2]):
        im1 = ants.from_numpy(fixed[:, :, slice])
        im2 = ants.from_numpy(moving[:, :, slice])
        reg12 = ants.registration(
            fixed=im1, moving=im2, type_of_transform='SyN')
        registered[:, :, slice] = reg12['warpedmovout'].numpy()
    return registered


def round_optimize_stage1(frames, tvec, rounds=3):
    T1s = []
    T1errs = []
    registereds = []
    update_Ms = []

    I = copy.copy(frames)

    for round in range(rounds):
        print(f"round {round}")
        if round == 0:
            tvec_r = np.array([tvec[0], tvec[-2], tvec[-1]])
            frames_r = np.dstack(
                [frames[..., 0], frames[..., -2], frames[..., -1]])
        else:
            tvec_r = tvec
            frames_r = registered_r
        inversion_recovery_img, T1, T1err = t1fitting_inter(
            frames_r, tvec_r, tvec)

        S = inversion_recovery_img
        M = np.abs(inversion_recovery_img)
        if round <= 3:
            step_size = -0.1
        else:
            step_size = -0.05
        update_M = synthetic(I, S, M, step_size=-0.01, iter=30)
        registered_r = register(update_M, I)
        T1s.append(T1)
        T1errs.append(T1err)
        registereds.append(registered_r)
        update_Ms.append(update_M)
    return T1s, T1errs, registereds, update_Ms


def round_optimize_stage2(frames, tvec, rounds=3):
    T1s = []
    T1errs = []
    registereds = []
    update_Ms = []

    I = copy.copy(frames)
    I[I >= 2000] = 0
    registered_r = copy.copy(frames)
    for round in range(rounds):
        print(f"round {round}")

        tvec_r = tvec
        frames_r = registered_r
        inversion_recovery_img, T1, T1err = t1fitting_intra(
            frames_r, tvec_r, tvec)

        S = inversion_recovery_img
        M = np.abs(inversion_recovery_img)
        if round <= 3:
            step_size = -0.01
        else:
            step_size = -0.01
        update_M = synthetic(I, S, M, step_size=step_size, iter=30)
        registered_r = register(update_M, I)
        T1s.append(T1)
        T1errs.append(T1err)
        registereds.append(registered_r)
        update_Ms.append(update_M)
    return T1s, T1errs, registereds, update_Ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 fitting')
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--minus', type=int,
                        help='minus value for acquisition time')
    parser.add_argument('--weight', type=bool, default=False)
    args = parser.parse_args()

    __file__ = args.input
    time_path = os.path.join(__file__, 'acquisitionTime.csv')
    ac_time_df = pd.read_csv(time_path)
    ac_time_dict = ac_time_df.to_dict()
    series = [int(value.split('_')[0][7:])
              for value in ac_time_dict['Subject'].values()]
    timeino = [value for value in ac_time_dict['AcquisitionTime'].values()]
    print(series, timeino)
    outputfolder = f"{__file__}/registration"
    shutil.rmtree(f"{outputfolder}/stage1", ignore_errors=True)
    shutil.rmtree(f"{outputfolder}/stage2", ignore_errors=True)
    os.makedirs(f"{outputfolder}/stage1", exist_ok=True)
    os.makedirs(f"{outputfolder}/stage2", exist_ok=True)
    if os.path.exists(f"{outputfolder}/register_1.npy"):
        revert_matrix = np.load(f"{outputfolder}/register_1.npy")
        revert_corr = np.load(f"{outputfolder}/register_2.npy")
    else:
        postT1w = {}
        if args.weight is True:
            key = 'PostconT1w'
        else:
            key = 'PostconT1'
        print(glob.glob(os.path.join(__file__, f'{key}/*')))
        for file in glob.glob(os.path.join(__file__, f'{key}/*')):
            scans = glob.glob(os.path.join(file, '*.IMA'))[0]
            subject = Path(file).stem
            subjectid = int(subject.split('_')[0][7:])
            print(subjectid)
            img = pydicom.dcmread(scans)
            postT1w[subjectid] = img.pixel_array

        postT1w_sorted = dict(sorted(postT1w.items()))
        img_arrays = np.dstack(list(postT1w_sorted.values()))
        print(img_arrays.shape)

        ac_time = np.squeeze(np.asarray(timeino))

        ac_idx = np.argsort(np.array(series))
        ac_time = ac_time[ac_idx]
        corr_t1 = img_arrays
        ac_time = ac_time - args.minus

        T1_intra, T1err_intra, registered_intra, update_M_intra = round_optimize_stage2(
            corr_t1.astype(np.float32), ac_time.astype(np.uint16), rounds=3)

        revert_matrix = np.zeros_like(registered_intra[2])
        revert_corr = np.zeros_like(registered_intra[2])
        revert_matrix = registered_intra[2]
        revert_corr = corr_t1

        revert_corr[revert_corr >= 2000] = 0

        np.save(f"{outputfolder}/register_1.npy", revert_matrix)
        np.save(f"{outputfolder}/register_2.npy", revert_corr)
    sitk.Show(sitk.GetImageFromArray(revert_corr.transpose(2, 0, 1)), 'corr')
    sitk.Show(sitk.GetImageFromArray(
            revert_matrix.transpose(2, 0, 1)), 'corr2')
    # print(files)
    for file in glob.glob(os.path.join(__file__, 'PostconT1/*')):
        scans = glob.glob(os.path.join(file, '*.IMA'))[0]
        subject = Path(file).stem

        img = pydicom.dcmread(scans)
        idx = int(subject.split('_')[0][7:])
        iddx = sorted(series).index(idx)
        print(subject, idx, iddx)
        data = revert_matrix[:, :, iddx]
        img_data = img.pixel_array
        img_data[:, :] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage1/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage1/{subject}", f"{subject}" + '.dcm'))
        # del img

        data = revert_corr[:, :, iddx]
        img = pydicom.dcmread(scans)
        img.PixelData = data.tobytes()
        os.makedirs(f"{outputfolder}/stage2/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage2/{subject}", f"{subject}" + '.dcm'))
    # del img
