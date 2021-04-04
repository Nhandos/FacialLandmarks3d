import glob
import json
import os
import logging
import argparse
import sys
from typing import Iterable, Tuple

import numpy as np
import cv2

CHESSBOARD_SIZE = (6, 9)
FIND_POINTS_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + \
                    cv2.CALIB_CB_FAST_CHECK + \
                    cv2.CALIB_CB_NORMALIZE_IMAGE

POINTS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

CALIBRATION_FLAGS = 0
#CALIBRATION_FLAGS += cv2.CALIB_USE_INTRINSIC_GUESS
#CALIBRATION_FLAGS += cv2.CALIB_FIX_PRINCIPAL_POINT 

STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

STEREO_FLAGS = 0
STEREO_FLAGS |= cv2.CALIB_FIX_INTRINSIC
#STEREO_FLAGS |= cv2.CALIB_FIX_PRINCIPAL_POINT
#STEREO_FLAGS += cv2.CALIB_USE_INTRINSIC_GUESS
#STEREO_FLAGS += cv2.CALIB_FIX_FOCAL_LENGTH
STEREO_FLAGS |= cv2.CALIB_FIX_ASPECT_RATIO
#STEREO_FLAGS += cv2.CALIB_ZERO_TANGENT_DIST
#STEREO_FLAGS += cv2.CALIB_RATIONAL_MODEL
#STEREO_FLAGS += cv2.CALIB_SAME_FOCAL_LENGTH
# STEREO_FLAGS += cv2.CALIB_FIX_K3
# STEREO_FLAGS += cv2.CALIB_FIX_K4
# STEREO_FLAGS += cv2.CALIB_FIX_K5


class StereoCalibrator(object):
    
    def __init__(self,
                 calibration_paths=None,
                 chessboard_size=CHESSBOARD_SIZE,
                 flags=FIND_POINTS_FLAGS):

        self.calib_paths = calibration_paths          
        self.chessboard_size = CHESSBOARD_SIZE
        self.flags = FIND_POINTS_FLAGS
        self.img_size = None
        
        # Calib. points
        self.obj_pts = []               # vector of vectors of calibration pattern points
        self.l_pts_arr = []             # vector of calibration pattern points from left view
        self.r_pts_arr = []             # vector of calibration pattern points from right view

        # Camera parametes
        self.l_k = None                 # left camera matrix
        self.l_d = None                 # left camera distortion coefficients
        self.r_k = None                 # right camera matrix
        self.r_d = None                 # right camera distortion coefficients

        # Stereo-rectification parameters
        self.R = None                   # Rotation matrix between coords. of 1st and 2nd camera
        self.T = None                   # Translation vector between coords. of 1st and 2nd camera
        self.E = None                   # Essential matrix
        self.F = None                   # Fundamental matrix
        self.l_rect = None              # left rectification transform matrix 
        self.r_rect = None              # right rectification transform matrix
        self.l_p = None                 # left projection matrix
        self.r_p = None                 # right projection matrix
        
        # Undistortion mapping matrix
        self.l_mapx = None                             
        self.l_mapy = None
        self.r_mapx = None
        self.r_mapy = None
        
    def find_calib_pts(self):
        
        assert len(self.calib_paths) > 0, "Zero calibration image pairs"
        _n_ok = 0
        
        # Create a meshgrid of points for the pattern coordinate
        _objp = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        _objp[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)

        for l_path, r_path in self.calib_paths:

            l_img, r_img = cv2.imread(l_path, 0), cv2.imread(r_path, 0)
            if self.img_size is None: self.img_size = l_img.shape
            assert self.img_size == l_img.shape == r_img.shape, "Calibration images size mismatch"

            l_ret, l_pts = cv2.findChessboardCorners(l_img, self.chessboard_size, None)
            r_ret, r_pts = cv2.findChessboardCorners(r_img, self.chessboard_size, None)

            if not (l_ret and r_ret):
                logging.info("Cannot find points in {}, {}".format(l_path, r_path))
                continue


            _n_ok += 1

            # Refine corners
            l_pts = cv2.cornerSubPix(l_img, l_pts, (11, 11), (-1, -1), POINTS_CRITERIA)
            r_pts = cv2.cornerSubPix(r_img, r_pts, (11, 11), (-1, -1), POINTS_CRITERIA)
            
            cv2.drawChessboardCorners(l_img, self.chessboard_size, l_pts, True)
            cv2.drawChessboardCorners(r_img, self.chessboard_size, r_pts, True)

            cv2.imshow('chessboard', np.hstack((l_img, r_img)))
            cv2.waitKey()

            # Append to list of good corners
            self.obj_pts.append(_objp)
            self.l_pts_arr.append(l_pts)
            self.r_pts_arr.append(r_pts)

        cv2.destroyAllWindows()
        logging.info("found {} stereo pairs for calibration".format(_n_ok))


    def save_calib_pts(self, outdir: str):
        """ Save all points """

        assert os.path.isdir(outdir), "Output director does not exists"
        assert len(self.obj_pts) > 0, "Zero calibration points"

        _objpts = np.array(self.obj_pts)
        _lpts_arr = np.array(self.l_pts_arr)
        _rpts_arr = np.array(self.rpts_arr)

        np.save(os.path.join(outdir, 'objpts'), _objpts, allow_pickle=False)
        np.save(os.path.join(outdir, 'l_pts_arr'), _lpts_arr, allow_pickle=False)
        np.save(os.path.join(outdir, 'r_pts_arr'), _rpts_arr, allow_pickle=False)

    def save_camera_params(self, outpath: str):
        
        if not outpath.endswith('.json'): outpath += '.json'

        with open(outpath, 'w+') as fp:
            json.dump({
                        'Dim': self.img_size,
                        'L_K': self.l_k.tolist(),
                        'L_D': self.l_d.tolist(),
                        'R_K': self.r_k.tolist(),
                        'R_D': self.r_d.tolist(),
                        'R': self.R.tolist(),
                        'T': self.T.tolist(),
                        'E': self.E.tolist(),
                        'F': self.F.tolist()
                      }, fp)

    def load_camera_params(self, path: str):

        with open(path, 'r') as fp:
            param = json.load(fp)
            self.img_size, self.l_k, self.l_d, self.r_k, self.r_d, self.R, self.T, self.E, self.F = \
                map(lambda x: np.array(x, np.float64), param.values())
            self.img_size = tuple(map(int, self.img_size.tolist()))
           
    def calc_camera_intrinsic(self) -> Tuple[Tuple[np.ndarray]]:

        assert len(self.obj_pts) > 0, "Zero calibration points"
        _, self.l_k, self.l_d, _, _ = cv2.calibrateCamera(self.obj_pts, self.l_pts_arr, self.img_size,
                                            CALIBRATION_FLAGS, None)
        _, self.r_k, self.r_d, _, _ = cv2.calibrateCamera(self.obj_pts, self.r_pts_arr,  self.img_size,
                                            CALIBRATION_FLAGS, None)

        return (self.l_k, self.l_d), (self.r_k, self.r_d)

    def stereo_calibrate(self): 

        ret, self.l_k, self.l_d, self.r_k, self.r_d, self.R, self.T, self.E, self.F = \
            cv2.stereoCalibrate(self.obj_pts,
                                self.l_pts_arr,
                                self.r_pts_arr,
                                self.l_k,
                                self.l_d,
                                self.r_k,
                                self.r_d,
                                self.img_size,
                                self.R,
                                self.T,
                                self.E,
                                self.F,
                                criteria=STEREO_CRITERIA,
                                flags=STEREO_FLAGS)

        if not ret:       
            logging.debug("Stereo calibration failed")
            raise ValueError

    def rectify(self, l_frame: np.ndarray, r_frame: np.ndarray):

        if self.l_rect is None:
            logging.info("Initialising undistortion maps")

            self.l_rect, self.r_rect, self.l_p, self.r_p, _, _, _ = \
                cv2.stereoRectify(self.l_k,
                                  self.l_d, 
                                  self.r_k,
                                  self.r_d,
                                  self.img_size,
                                  self.R,
                                  self.T,
                                  0,
                                  (0, 0))

            
            self.l_mapx, self.l_mapy = cv2.initUndistortRectifyMap(self.l_k,
                                                                   self.l_d,
                                                                   self.l_rect,
                                                                   self.l_p,
                                                                   self.img_size,
                                                                   cv2.CV_16SC2)

            self.r_mapx, self.r_mapy = cv2.initUndistortRectifyMap(self.r_k,
                                                                   self.r_d,
                                                                   self.r_rect,
                                                                   self.r_p,
                                                                   self.img_size,
                                                                   cv2.CV_16SC2)
        
        return (cv2.remap(l_frame, self.l_mapx, self.l_mapy, cv2.INTER_AREA),
            cv2.remap(r_frame, self.r_mapx, self.r_mapy, cv2.INTER_AREA))
    


    def params_tostring(self):
        
        _break = "=" * 70
        str_ = '\n'.join([_break,
                          "<Dimension>",
                          str(self.img_size),
                          "\n<Left Camera Matrix>",            
                          str(self.l_k),
                          "\n<Left Camera Distortion Coefficients>",
                          str(self.l_d),
                          "\n<Right Camera Matrix>",
                          str(self.r_k),
                          "\n<Right Camera Distortion Coefficients>",
                          str(self.r_d),
                          "\n<Rotation Matrix>",
                          str(self.R),
                          "\n<Translation Vector>",
                          str(self.T),
                          "\n<Essential Matrix>",
                          str(self.E),
                          "\n<Fundamental Matrix>",
                          str(self.F),
                          _break,
                          ])

        return str_
