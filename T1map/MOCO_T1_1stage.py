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

def plot_T1s(imgs):
    length = len(imgs)
    fig, axs = plt.subplots(1, length, figsize=(length*5, 5))
    for idx, img in enumerate(imgs):
        axs[idx].imshow(img, vmin=0, vmax=1000)
        axs[idx].set_title(f"T1 map {idx}")
        axs[idx].axis('off') 
    return fig   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 fitting')
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--output', type=str, help='output folder')
    parser.add_argument('--clean', type=bool, default=False)
    parser.add_argument('--threshold', type=int, default=2000)
    args = parser.parse_args()

    __input_base__ = args.input
    __output_base__ = args.output
    threshold = args.threshold
    # threshold = 2000
    # __input_base__ = '/Users/mona/Library/CloudStorage/Box-Box/Pig_acuteScans_dDCE/Pig_2326_Baseline'
    # __output_base__ = '/Users/mona/Library/CloudStorage/Box-Box/Pig_acuteScans_dDCE/Pig_2326_Baseline_T1Map_Mona'
    os.makedirs(__output_base__, exist_ok=True)

    rang = 96 // 2

    subjects = sorted(glob.glob(os.path.join(__input_base__, 'T1MAP*')))
    for file in subjects:
        if 'MOCO' not in file:
            print(f"{Path(file).name} in processing")
            output_folder = f"{__output_base__}/{Path(file).name}_Mona"
            # if args.clean and os.path.exists(output_folder):
            #     shutil.rmtree(output_folder, ignore_errors=True)

            T1w_scans = []
            tvec = []

            if os.path.exists(output_folder):
                print(f"{file} already processed")
            else:
                scans = sorted(glob.glob(os.path.join(file, '*.IMA')))

                raw_imgs = []
                for scan in scans:
                    img = pydicom.dcmread(scan)
                    subject = Path(scan).stem
                    T1w_scans.append(img.pixel_array)
                    tvec.append(float(img.InversionTime))
                    print(f"The {subject} has inversion time {img.InversionTime}")
                    raw_imgs.append((img, subject))

                tvec = np.array(tvec)
                tvec_idx = np.argsort(tvec)
                tvec = tvec[tvec_idx]
                orig_frames = np.dstack(T1w_scans)
                orig_frames = orig_frames[:, :, tvec_idx]
                print(f"Original image size {orig_frames.shape}")
                x = orig_frames.shape[0]//2
                y = orig_frames.shape[1]//2
                orig_frames = orig_frames[x-rang:x+rang, y-rang:y+rang, :]
                T1s, T1errs, registereds, update_Ms = round_optimize_stage1(
                    orig_frames.astype(np.float32), tvec, threshold=threshold, rounds=3)

                T1 = np.abs(T1s[-1])
                registered_map = registereds[-1]
                print(f"T1 shape {T1.shape} and registered map shape {registered_map.shape}")

                # save the registered image
                os.makedirs(output_folder, exist_ok=True)

                for idx, item in enumerate(raw_imgs):
                    img, subject = item
                    data = registered_map[..., tvec_idx[idx]]
                    img_data = img.pixel_array
                    x = img_data.shape[0]//2
                    y = img_data.shape[1]//2
                    img_data[x-rang:x+rang, y-rang:y+rang] = data
                    img.PixelData = img_data.tobytes()
                    
                    img.save_as(f"{output_folder}/{idx}.IMA")
                
                # save the T1s
                output_folder_T1 = f"{__output_base__}/{Path(file).name}_T1_Mona"
                os.makedirs(output_folder_T1, exist_ok=True)
                img, subject = raw_imgs[0]
                data = registered_map[..., tvec_idx[idx]]
                img_data = img.pixel_array
                x = img_data.shape[0]//2
                y = img_data.shape[1]//2
                img_data[x-rang:x+rang, y-rang:y+rang] = data
                img.PixelData = img_data.tobytes()
                img.save_as(f"{output_folder_T1}/{idx}.IMA")

                fig = plot_T1s(T1s)
                fig.savefig(f"{output_folder_T1}/T1s.png")
