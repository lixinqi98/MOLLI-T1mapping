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
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--dual', type=bool, default=False,
                        help='True for the use of dual exponential model')
    parser.add_argument('--weight', type=bool, default=False)
    parser.add_argument('--threshold', type=int, default=2000)
    parser.add_argument('--minus', type=int, default=44000,
                        help='minus value for acquisition time')
    parser.add_argument('--clean', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Use ImgShow to visualize the results')
    args = parser.parse_args()

    # __file__ = args.input
    __file__ = '/Users/mona/Documents/data/registration/patient_T1'
    time_path = os.path.join(__file__, 'acquisitionTime_2_5_8.csv')
    ac_time_df = pd.read_csv(time_path)
    ac_time_dict = ac_time_df.to_dict()
    series = [int(re.findall(r'T1_\d+', value)[0][3:])
              for value in ac_time_dict['Subject'].values()]
    timeino = [value for value in ac_time_dict['AcquisitionTime'].values()]
    print(series, timeino)
    if not args.dual:
        outputfolder = f"{__file__}/registration/single_exp"
    else:
        print(f"Use the dual exponential model, {args.dual}")
        outputfolder = f"{__file__}/registration/dual_exp"
    
    if args.clean:
        shutil.rmtree(outputfolder, ignore_errors=True)

    shutil.rmtree(f"{outputfolder}/stage1", ignore_errors=True)
    shutil.rmtree(f"{outputfolder}/stage2", ignore_errors=True)
    os.makedirs(f"{outputfolder}/stage1", exist_ok=True)
    os.makedirs(f"{outputfolder}/stage2", exist_ok=True)

    rang = 128 // 2
    key = "PostconT1_2_5_8"
    if os.path.exists(f"{outputfolder}/register_stage1.npy"):
        print("Load the registered images")
        revert_stage1 = np.load(f"{outputfolder}/register_stage1.npy")
        revert_stage2 = np.load(f"{outputfolder}/register_stage2.npy")
    else:
        postT1w = {}
        
        print(glob.glob(os.path.join(__file__, f'{key}/*')))
        for file in glob.glob(os.path.join(__file__, f'{key}/*')):
            scans = glob.glob(os.path.join(file, '*.dcm'))[0]
            subject = Path(file).stem
            subjectid = int(re.findall(r'T1_\d+', subject)[0][3:])
            print(subjectid)
            img = pydicom.dcmread(scans)
            postT1w[subjectid] = img.pixel_array

        postT1w_sorted = dict(sorted(postT1w.items()))
        orig_frames = np.dstack(list(postT1w_sorted.values()))  # orig_frames is already sorted based on the key
        # cropped the images at the center
        if args.verbose:
            sitk.Show(sitk.GetImageFromArray(
                orig_frames.transpose(2, 0, 1)), 'Original Image')
            print(f"Original image size {orig_frames.shape}")
        x = orig_frames.shape[0]//2
        y = orig_frames.shape[1]//2
        cropped_frames = orig_frames[x-rang:x+rang, y-rang:y+rang, :]
        if args.verbose:
            sitk.Show(sitk.GetImageFromArray(
                cropped_frames.transpose(2, 0, 1)), 'Cropped Image')
            print(f"Cropped image size {cropped_frames.shape}")

        ac_time = np.squeeze(np.asarray(timeino))
        ac_idx = np.argsort(np.array(series))
        ac_time = ac_time[ac_idx]

        ac_time = ac_time - args.minus

        cropped_frames[cropped_frames >= args.threshold] = 0

        T1_intra, T1err_intra, registered_intra, update_M_intra = round_optimize_stage2(
            cropped_frames.astype(np.float32), ac_time.astype(np.uint16), 
            dual=args.dual, 
            step_size=0.005,
            threshold=args.threshold, 
            rounds=3)

        matrix_stage1 = cropped_frames
        matrix_stage2 = registered_intra[-1]

        matrix_stage1[matrix_stage1 >= args.threshold] = 0
        matrix_stage2[matrix_stage2 >= args.threshold] = 0

        np.save(f"{outputfolder}/register_stage1.npy", matrix_stage1)
        np.save(f"{outputfolder}/register_stage2.npy", matrix_stage1)

        if args.verbose:
            sitk.Show(sitk.GetImageFromArray(
                matrix_stage1.transpose(2, 0, 1)), 'Stage1')
            sitk.Show(sitk.GetImageFromArray(
                matrix_stage2.transpose(2, 0, 1)), 'Stage2')
            sitk.Show(sitk.GetImageFromArray(
                (matrix_stage2 - matrix_stage1).transpose(2, 0, 1)), 'Diff')

    for file in glob.glob(os.path.join(__file__, f'{key}/*')):
        scans = glob.glob(os.path.join(file, '*.dcm'))[0]
        subject = Path(file).stem

        img = pydicom.dcmread(scans)
        idx = int(re.findall(r'T1_\d+', subject)[0][3:])
        iddx = sorted(series).index(idx)
        print(subject, idx, iddx)
        data = matrix_stage1[:, :, iddx]
        img_data = img.pixel_array
        x = img_data.shape[0]//2
        y = img_data.shape[1]//2
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage1/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage1/{subject}", f"{subject}" + '.dcm'))
        # del img

        data = matrix_stage2[:, :, iddx]
        img = pydicom.dcmread(scans)
        img_data = img.pixel_array
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage2/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage2/{subject}", f"{subject}" + '.dcm'))

