import argparse
import glob
import os
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from utils import *

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 fitting')
    parser.add_argument('--input', type=str, default='/Users/mona/Library/CloudStorage/Box-Box/Data/Structured data/Animals/Paris/Paris_D8', help='input folder')
    parser.add_argument('--dual', type=bool, default=True,
                        help='True for the use of dual exponential model')
    parser.add_argument('--weight', type=bool, default=False)
    parser.add_argument('--threshold', type=int, default=2000)
    args = parser.parse_args()

    __file__ = args.input
    time_path = os.path.join(__file__, 'acquisitionTime.csv')
    ac_time_df = pd.read_csv(time_path)
    ac_time_dict = ac_time_df.to_dict()
    series = [int(value.split('_')[0][7:])
              for value in ac_time_dict['Subject'].values()]
    timeino = [value for value in ac_time_dict['AcquisitionTime'].values()]
    print(series, timeino)
    if not args.dual:
        outputfolder = f"{__file__}/registration/single_exp"
    else:
        print(f"Use the dual exponential model, {args.dual}")
        outputfolder = f"{__file__}/registration/dual_exp"
    shutil.rmtree(f"{outputfolder}/stage1", ignore_errors=True)
    shutil.rmtree(f"{outputfolder}/stage2", ignore_errors=True)
    os.makedirs(f"{outputfolder}/stage1", exist_ok=True)
    os.makedirs(f"{outputfolder}/stage2", exist_ok=True)

    rang = 96 // 2

    if os.path.exists(f"{outputfolder}/register_1.npy"):
        revert_stage1 = np.load(f"{outputfolder}/register_1.npy")
        revert_stage2 = np.load(f"{outputfolder}/register_2.npy")
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
        orig_img_arrays = np.dstack(list(postT1w_sorted.values()))
        # cropped the images at the center
        # sitk.Show(sitk.GetImageFromArray(
        #     orig_img_arrays.transpose(2, 0, 1)), 'Original Size Image')
        # print(f"Original image size {orig_img_arrays.shape}")

        img_arrays = orig_img_arrays[orig_img_arrays.shape[0]//2-rang:orig_img_arrays.shape[0] //
                                     2+rang, orig_img_arrays.shape[1]//2-rang:orig_img_arrays.shape[1]//2+rang, :]
        # sitk.Show(sitk.GetImageFromArray(
        #     img_arrays.transpose(2, 0, 1)), 'Cropped Size Image')
        # print(f"Cropped image size {img_arrays.shape}")

        ac_time = np.squeeze(np.asarray(timeino))

        ac_idx = np.argsort(np.array(series))
        ac_time = ac_time[ac_idx]
        corr_t1 = img_arrays
        ac_time = ac_time - np.min(ac_time)

        T1_intra, T1err_intra, registered_intra, update_M_intra = round_optimize_stage2(
            corr_t1.astype(np.float32), ac_time.astype(np.uint16), dual=args.dual, threshold=2000, rounds=3)

        revert_matrix = np.zeros_like(registered_intra[2])
        revert_corr = np.zeros_like(registered_intra[2])
        revert_stage2 = registered_intra[2]
        revert_stage1 = corr_t1

        # revert_stage1[revert_stage1 >= 2000] = 0
        # revert_stage2[revert_stage2 >= 2000] = 0

        np.save(f"{outputfolder}/register_1.npy", revert_stage1)
        np.save(f"{outputfolder}/register_2.npy", revert_stage2)
    # sitk.Show(sitk.GetImageFromArray(
    #     revert_stage1.transpose(2, 0, 1)), 'Stage1')
    # sitk.Show(sitk.GetImageFromArray(
    #     revert_stage2.transpose(2, 0, 1)), 'Stage2')
    # sitk.Show(sitk.GetImageFromArray(
    #     (revert_stage2 - revert_stage1).transpose(2, 0, 1)), 'Diff')

    # print(files)
    for file in glob.glob(os.path.join(__file__, 'PostconT1/*')):
        scans = glob.glob(os.path.join(file, '*.IMA'))[0]
        subject = Path(file).stem

        img = pydicom.dcmread(scans)
        idx = int(re.findall(r'\d+', subject)[0])
        iddx = sorted(series).index(idx)
        print(subject, idx, iddx)
        data = revert_stage1[:, :, iddx]
        img_data = img.pixel_array
        x = img_data.shape[0]//2
        y = img_data.shape[1]//2
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage1/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage1/{subject}", f"{subject}" + '.dcm'))
        # del img

        data = revert_stage2[:, :, iddx]
        img = pydicom.dcmread(scans)
        img_data = img.pixel_array
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage2/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage2/{subject}", f"{subject}" + '.dcm'))
    # del img
