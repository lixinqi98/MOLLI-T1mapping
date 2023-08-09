import glob
import os
import argparse
from pathlib import Path
import re
import numpy as np
import pydicom
import SimpleITK as sitk


parser = argparse.ArgumentParser(description='T1 fitting')
parser.add_argument('--input', type=str, default='/Users/mona/Library/CloudStorage/Box-Box/Animals/Cinnemon/Cinnemon_D6', help='input folder')
parser.add_argument('--minus', type=int, default=53000,
                    help='minus value for acquisition time')
parser.add_argument('--weight', type=bool, default=True)

parser.add_argument('--threshold', type=int, default=2000)
args = parser.parse_args()

__file__ = args.input
outputfolder = f"{__file__}/registration"

T1_scans = {}
for file in glob.glob(os.path.join(__file__, 'PostconT1/POSTCON*')):
    scans = glob.glob(os.path.join(file, '*.IMA'))
    subject = Path(file).stem
    subjectid = int(re.findall(r'\d+', subject)[0])

    print(f"{subject} - {subjectid} in processing")
    
    tvec = []

    T1_path = f"{outputfolder}/stage1/{subjectid}_T1.npy"
    

    img = pydicom.dcmread(scans[0])
    T1_scans[subjectid] = img.pixel_array

    postT1w_sorted = dict(sorted(T1_scans.items()))
    img_arrays = np.dstack(list(postT1w_sorted.values()))
rang = 96 // 2
x = img_arrays.shape[0]//2
y = img_arrays.shape[1]//2
img_arrays = img_arrays[x-rang:x+rang, y-rang:y+rang, :]
sitk.WriteImage(sitk.GetImageFromArray(img_arrays.transpose(1, 2, 0)), f"{outputfolder}/POSTCONT1.nii")