import argparse
import copy
import glob
import os
import re
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from utils import *

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 fitting')
    parser.add_argument('--input', type=str, default='/Users/mona/Library/CloudStorage/Box-Box/Animals/Cinnemon/Cinnemon_D6', help='input folder')
    parser.add_argument('--minus', type=int, default=53000,
                        help='minus value for acquisition time')
    parser.add_argument('--weight', type=bool, default=True)

    parser.add_argument('--threshold', type=int, default=2000)
    args = parser.parse_args()

    __file__ = args.input

    time_path = os.path.join(__file__, 'acquisitionTime.csv')
    ac_time_df = pd.read_csv(time_path)
    ac_time_dict = ac_time_df.to_dict()
    series = [int(re.findall(r'\d+', value)[0])
              for value in ac_time_dict['Subject'].values()]
    timeino = list(ac_time_dict['AcquisitionTime'].values())
    timeino_idx = np.argsort(series)
    timeino = np.array(timeino)[timeino_idx]
    print(timeino, len(timeino))
    postT1w = {}

    outputfolder = f"{__file__}/registration"
    # shutil.rmtree(f"{outputfolder}/stage1", ignore_errors=True)
    # shutil.rmtree(f"{outputfolder}/stage2", ignore_errors=True)
    os.makedirs(f"{outputfolder}/stage1", exist_ok=True)
    os.makedirs(f"{outputfolder}/stage2", exist_ok=True)

    rang = 96 // 2
    if os.path.exists(f"{outputfolder}/register_1.npy"):
        revert_stage1 = np.load(f"{outputfolder}/register_1.npy")
        revert_stage2 = np.load(f"{outputfolder}/register_2.npy")
        sitk.WriteImage(sitk.GetImageFromArray(revert_stage1.transpose(1, 2, 0)), f"{outputfolder}/register_1.nii")
        sitk.WriteImage(sitk.GetImageFromArray(revert_stage2.transpose(1, 2, 0)), f"{outputfolder}/register_2.nii")
    else:

        for file in glob.glob(os.path.join(__file__, 'PostconT1w/POSTCON*')):
            scans = glob.glob(os.path.join(file, '*.IMA'))
            subject = Path(file).stem
            subjectid = int(re.findall(r'\d+', subject)[0])

            print(f"{subject} - {subjectid} in processing")
            T1w_scans = []
            tvec = []

            T1_path = f"{outputfolder}/stage1/{subjectid}_T1.npy"
            if os.path.exists(T1_path):
                postT1w[subjectid] = np.abs(np.load(T1_path))
                print(f"{subject} - {subjectid} already processed")
            else:
                if subjectid == 1:
                    print('ok')
                for scan in scans:
                    img = pydicom.dcmread(scan)
                    subject = Path(file).stem

                    T1w_scans.append(img.pixel_array)
                    tvec.append(float(img.InversionTime))
                tvec = np.array(tvec)
                tvec_idx = np.argsort(tvec)
                tvec = tvec[tvec_idx]
                orig_frames = np.dstack(T1w_scans)
                orig_frames = orig_frames[:, :, tvec_idx]
                print(f"Original image size {orig_frames.shape}")
                x = orig_frames.shape[0]//2
                y = orig_frames.shape[1]//2
                orig_frames = orig_frames[x-rang:x+rang, y-rang:y+rang, :]
                print(f"Cropped image size {orig_frames.shape}")
                T1s, T1errs, registereds, update_Ms = round_optimize_stage1(
                    orig_frames.astype(np.float32), tvec, threshold=args.threshold, rounds=3)

                postT1w[subjectid] = np.abs(T1s[-1])
                np.save(T1_path, postT1w[subjectid])
                print(f"{subject} - {subjectid} processed and save")

        postT1w_sorted = dict(sorted(postT1w.items()))
        img_arrays = np.dstack(list(postT1w_sorted.values()))
        img_arrays[img_arrays >= 2000] = 0

        ac_time = np.squeeze(np.asarray(timeino))

        corr_t1 = img_arrays
        ac_time = ac_time - args.minus

        T1_intra, T1err_intra, registered_intra, update_M_intra = round_optimize_stage2(
            corr_t1.astype(np.float32), ac_time.astype(np.uint16), args.threshold, rounds=2)

        revert_matrix = np.zeros_like(registered_intra[-1])
        revert_corr = np.zeros_like(registered_intra[-1])
        revert_stage2 = registered_intra[-1]
        revert_stage1 = corr_t1

        np.save(f"{outputfolder}/register_1.npy", revert_stage1)
        np.save(f"{outputfolder}/register_2.npy", revert_stage2)

    try:
        sitk.Show(sitk.GetImageFromArray(
            revert_stage1.transpose(2, 0, 1)), 'Stage1')
        sitk.Show(sitk.GetImageFromArray(
            revert_stage2.transpose(2, 0, 1)), 'Stage2')
        sitk.Show(sitk.GetImageFromArray(
            (revert_stage2 - revert_stage1).transpose(2, 0, 1)), 'Diff')
    except:
        print("No GUI available")

    # print(files)
    for file in glob.glob(os.path.join(__file__, 'PostconT1/*')):
        scans = glob.glob(os.path.join(file, '*.IMA'))[0]
        subject = Path(file).stem

        img = pydicom.dcmread(scans)
        idx = int(re.findall(r'\d+', subject)[0])
        iddx = sorted(series).index(idx)
        print(subject, idx, iddx)
        data = revert_stage1[:, :, idx-1]
        img_data = img.pixel_array
        x = img_data.shape[0]//2
        y = img_data.shape[1]//2
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage1/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage1/{subject}", f"{subject}" + '.dcm'))
        # del img

        data = revert_stage2[:, :, idx-1]
        img = pydicom.dcmread(scans)
        img_data = img.pixel_array
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage2/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage2/{subject}", f"{subject}" + '.dcm'))
