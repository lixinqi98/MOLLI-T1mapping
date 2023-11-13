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
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=RuntimeWarning)


# A logger for this file
hydralog = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config_phase2")
def main(cfg: DictConfig):
    conf = OmegaConf.structured(OmegaConf.to_container(cfg, resolve=True))
    hydralog.info(f"Conf: {conf}")


    __file__ = conf.input
    # __file__ = '/Users/mona/Library/CloudStorage/Box-Box/Mona/patient_registration_LiTing_Mona/015'
    time_path = os.path.join(__file__, 'acquisitionTime.csv')
    ac_time_df = pd.read_csv(time_path)
    ac_time_dict = ac_time_df.set_index('Subject').to_dict()['AcquisitionTime']

    key = "PostconT1"
    label = conf.label
    post_key = f"T1_*{label}*"

    isDual = False
    if conf.exp == "single":
        hydralog.info(f"Use the single exponential model")
        outputfolder = f"{__file__}/registration/{label}/single_exp"
    elif conf.exp == "dual":
        isDual = True
        hydralog.info(f"Use the dual exponential model")
        outputfolder = f"{__file__}/registration/{label}/dual_exp"
    else:
        raise ValueError("The experiment type is not defined")
    
    if conf.clean:
        hydralog.info(f"Clean the folder {outputfolder}")
        shutil.rmtree(outputfolder, ignore_errors=True)

    os.makedirs(f"{outputfolder}/stage1", exist_ok=True)
    os.makedirs(f"{outputfolder}/stage2", exist_ok=True)

    rang = conf.FOV // 2
    

    postT1w = {}
    
    hydralog.debug(glob.glob(os.path.join(__file__, f'{key}/{post_key}')))
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
    if conf.verbose:
        sitk.Show(sitk.GetImageFromArray(
            orig_frames.transpose(2, 0, 1)), 'Original Image')
        hydralog.debug(f"Original image size {orig_frames.shape}")
    x = orig_frames.shape[0]//2
    y = orig_frames.shape[1]//2
    cropped_frames = orig_frames[x-rang:x+rang, y-rang:y+rang, :]
    if conf.verbose:
        sitk.Show(sitk.GetImageFromArray(
            cropped_frames.transpose(2, 0, 1)), 'Cropped Image')
        hydralog.debug(f"Cropped image size {cropped_frames.shape}")

    ac_time = np.array(ordered_times) - conf.base_actime

    if os.path.exists(f"{outputfolder}/register_stage1.npy"):
        hydralog.info("Load the pre-registered images")
        matrix_stage1 = np.load(f"{outputfolder}/register_stage1.npy")
        matrix_stage2 = np.load(f"{outputfolder}/register_stage2.npy")
    else:

        cropped_frames[cropped_frames >= conf.threshold] = 0

        T1_intra, T1err_intra, registered_intra, update_M_intra = round_optimize_stage2(
            cropped_frames.astype(np.float32), ac_time.astype(np.uint16), 
            dual=isDual, 
            step_size=0.005,
            threshold=conf.threshold, 
            rounds=3)

        matrix_stage1 = cropped_frames
        matrix_stage2 = registered_intra[-1]

        matrix_stage1[matrix_stage1 >= conf.threshold] = 0
        matrix_stage2[matrix_stage2 >= conf.threshold] = 0

        np.save(f"{outputfolder}/register_stage1.npy", matrix_stage1)
        np.save(f"{outputfolder}/register_stage2.npy", matrix_stage1)

    if conf.verbose:
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
    
        hydralog.info(f"Save the subject {subject}, index {idx}")


if __name__ == "__main__":
    main()