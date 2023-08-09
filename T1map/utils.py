import numpy as np
import copy
import ants

import scipy.io as sio
import SimpleITK as sitk


import calc_t1map_inter as t1_inter
import calc_t1map_intra as t1_intra
import matplotlib.pyplot as plt

from Optimize import Optimize


def round_optimize_stage1(frames, tvec, threshold=2000, rounds=3):
    T1s = []
    T1errs = []
    registereds = []
    update_Ms = []

    I = copy.copy(frames)

    for round in range(rounds):
        print(f"round {round}")
        if round == 0:
            tvec_r = np.array([tvec[0], tvec[1], tvec[-1]])
            frames_r = np.dstack(
                [frames[..., 0], frames[..., 1], frames[..., -1]])
        else:
            tvec_r = tvec
            frames_r = registered_r
        inversion_recovery_img, T1, T1err = t1fitting_inter(
            frames_r, tvec_r, tvec)
        
        # for images where t1 < 0, which means bad t1 fitting, change back to original value
        mask = copy.deepcopy(T1)
        mask[mask > 0] = 0
        mask[mask < 0] = 1
        mask = mask.astype('bool')
        if round == 0:
            inversion_recovery_img[mask, :] = frames[mask, :]
        else:
            inversion_recovery_img[mask, :] = registered_r[mask, :]

        S = inversion_recovery_img
        M = np.abs(inversion_recovery_img)
        if round <= 2:
            step_size = -0.1
        else:
            step_size = -0.05
        update_M = synthetic(I, S, M, threshold, step_size=-0.01, iter=30)
        registered_r = register(update_M, I)
        T1s.append(T1)
        T1errs.append(T1err)
        registereds.append(registered_r)
        update_Ms.append(update_M)
    return T1s, T1errs, registereds, update_Ms


def round_optimize_stage2(frames, tvec, step_size=0.01, threshold=2000, rounds=3):
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
        
        update_M = synthetic(I, S, M, threshold, step_size=step_size, iter=30)
        registered_r = register(update_M, I)
        T1s.append(T1)
        T1errs.append(T1err)
        registereds.append(registered_r)
        update_Ms.append(update_M)
    return T1s, T1errs, registereds, update_Ms


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
    b = t1_params_pre[:, :, 1] + 1e-10
    c = t1_params_pre[:, :, 2]
    t1 = (1 / b) * (a / (a + c) - 1)
    # t1 = a * (1 - np.exp(-x / b)) + c

    inversion_recovery_img = np.zeros(
        (ini_frames.shape[0], ini_frames.shape[1], tvec.shape[-1]))
    for i in range(tvec.shape[-1]):
        x = tvec[i]
        inversion_recovery_img[..., i] = a * (1 - np.exp(-(x / b))) + c
    return inversion_recovery_img, t1, sqerrormap


def synthetic(I, S, M, threshold, step_size=-0.1, iter=3, alpha=1, beta=1):
    opt = Optimize(threshold)
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
    threshold = 500
    for slice in range(fixed.shape[2]):
        
        diff = moving[:, :, slice] - fixed[:, :, slice]
        mask = np.abs(diff)
        mask[mask < threshold] = 1
        mask[mask >= threshold] = 0
        mask = mask.astype('bool')
        im1 = ants.from_numpy(fixed[:, :, slice] * mask)
        im2 = ants.from_numpy(moving[:, :, slice] * mask)
        reg12 = ants.registration(
            fixed=im1, moving=im2, type_of_transform='SyN')
        registered[:, :, slice] = reg12['warpedmovout'].numpy()
    return registered