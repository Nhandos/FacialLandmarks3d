#!/usr/bin/env python3
""" Performs stereo calibration using chessboard """

import argparse
import logging
import sys
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
import cv2

from stereo import StereoCalibrator

root = logging.getLogger()
root.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # Suppress matplotlib debug
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctimes)s - %(name)s - %(levelname)s - %(message)s')
root.addHandler(handler)


def _main(_argv):

    img_paths = list(zip(glob.glob(_argv.left), glob.glob(_argv.right)))
    calibrator = StereoCalibrator(calibration_paths=img_paths)
    
    # Perform calibration & stereo rectification
    logging.info("Detecting calibration points...")
    calibrator.find_calib_pts()

    logging.info("Calculating camera intrinsic parameters...")
    calibrator.calc_camera_intrinsic()
    logging.info("Performing stereo calibiration...")
    calibrator.stereo_calibrate()
    logging.info(calibrator.params_tostring())

    logging.info("Saving parameters...")   
    calibrator.save_camera_params(_argv.out)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Stereo camera calibrator")
    parser.add_argument('--left' , type=str, required=True,
        help='director to left images')
    parser.add_argument('--right' , type=str, required=True,
        help='director to right images')
    parser.add_argument('--out', type=str, required=True,
        help='output calibration parameters path')
    
    _main(parser.parse_args())