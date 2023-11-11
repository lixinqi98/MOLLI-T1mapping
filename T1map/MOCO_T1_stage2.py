import argparse
import glob
import os
import re
import shutil
import warnings
from pathlib import Path
from collections import OrderedDict
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
    parser.add_argument('--clean', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Use ImgShow to visualize the results')
    parser.add_argument('--label', type=str, default='CE9')
    args = parser.parse_args()

    # __file__ = args.input
    __file__ = '/Users/mona/Library/CloudStorage/GoogleDrive-xinqili16@g.ucla.edu/My Drive/Registration/patient_Liting_dDCE_Mona/017'
    time_path = os.path.join(__file__, 'acquisitionTime.csv')
    ac_time_df = pd.read_csv(time_path)
    ac_time_dict = ac_time_df.set_index('Subject').to_dict()['AcquisitionTime']

    key = "PostconT1"
    label = args.label
    post_key = f"T1_*{label}*"

    if not args.dual:
        outputfolder = f"{__file__}/registration/{label}/single_exp"
    else:
        print(f"Use the dual exponential model, {args.dual}")
        outputfolder = f"{__file__}/registration/{label}/dual_exp"
    
    if args.clean:
        shutil.rmtree(outputfolder, ignore_errors=True)

        shutil.rmtree(f"{outputfolder}/stage1", ignore_errors=True)
        shutil.rmtree(f"{outputfolder}/stage2", ignore_errors=True)
        os.makedirs(f"{outputfolder}/stage1", exist_ok=True)
        os.makedirs(f"{outputfolder}/stage2", exist_ok=True)

    rang = 128 // 2
    

    postT1w = {}
    
    print(glob.glob(os.path.join(__file__, f'{key}/{post_key}')))
    for file in glob.glob(os.path.join(__file__, f'{key}/{post_key}')):
        scans = glob.glob(os.path.join(file, '*.dcm'))[0]
        subject = Path(file).stem
        img = pydicom.dcmread(scans)
        timeino = ac_time_dict[subject]
        postT1w[subject] = (img.pixel_array, int(timeino), scans)

    postT1w = OrderedDict(sorted(postT1w.items(), key=lambda t: t[1][1]))
    ordered_frames = [img for img, _, _ in postT1w.values()]
    ordered_times = [time for _, time, _ in postT1w.values()]
    orig_frames = np.dstack(ordered_frames)  # orig_frames is already sorted based on the key
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

    ac_time = np.array(ordered_times)

    ac_time = ac_time - args.minus

    if os.path.exists(f"{outputfolder}/register_stage1.npy"):
        print("Load the registered images")
        matrix_stage1 = np.load(f"{outputfolder}/register_stage1.npy")
        matrix_stage2 = np.load(f"{outputfolder}/register_stage2.npy")
    else:

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

    # for file in glob.glob(os.path.join(__file__, f'{key}/{post_key}')):

    for idx, subject in enumerate(postT1w.keys()):
        scans = postT1w[subject][2]
        img = pydicom.dcmread(scans)

        data = matrix_stage1[:, :, idx]
        img_data = img.pixel_array
        x = img_data.shape[0]//2
        y = img_data.shape[1]//2
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage1/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage1/{subject}", f"{subject}" + '.dcm'))
        # del img

        data = matrix_stage2[:, :, idx]
        img = pydicom.dcmread(scans)
        img_data = img.pixel_array
        img_data[x-rang:x+rang, y-rang:y+rang] = data
        img.PixelData = img_data.tobytes()
        os.makedirs(f"{outputfolder}/stage2/{subject}", exist_ok=True)
        img.save_as(os.path.join(
            f"{outputfolder}/stage2/{subject}", f"{subject}" + '.dcm'))
    
        print(f"Save the subject {subject}, index {idx}")

