from typing import Tuple

import cv2
import numpy as np


def detect_face(image: np.ndarray):
    # detect face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    return faces[-1]  # todo: return the largest ROI face


def extract_feature(image: np.ndarray, roi: Tuple[int, int, int, int]):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if roi is not None:
        # mask roi
        x, y, w, h = roi
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y:y+h, x:x+w, ...] = gray[y:y+h, x:x+w, ...]

        # image processing
        mask = cv2.equalizeHist(mask)
    else:
        mask = gray

    # features detection + extraction 
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(mask, 3000,  
        qualityLevel=0.01, minDistance=7)
    
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(image, kps)

    return kps, des


def match_features(frame1, frame2, normtype=cv2.NORM_HAMMING, loweratio=0.75):
    bfmatcher = cv2.BFMatcher(normtype, crossCheck=False)

    kp1, des1 = frame1.keypoints, frame1.descriptor
    kp2, des2 = frame2.keypoints, frame2.descriptor

    # match kps
    matches = bfmatcher.knnMatch(des1, des2, 2) # requires atleast 2 nearest matches

    # loweratio filter
    good = []
    for i, knnmatch in enumerate(matches):
        m, n = knnmatch[:2]
        if m.distance < loweratio * n.distance and m.distance < 32:
            good.append(m)

    return good


class Frame(object):


    def __init__(self, image: np.ndarray):

        self.image = image 

        if self.image is not None:
            self.roi = detect_face(self.image)
            self._kps, self._des = extract_feature(self.image, self.roi)

    def getAnnotated(self, matches=None):
        
        # keypoints labelling
        ret = cv2.drawKeypoints(self.image, self.keypoints, None, color=(0, 255, 0), 
            flags=0)

        # ROI
        if self.roi is not None:
            x, y, w, h = self.roi
            ret = cv2.rectangle(ret, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return ret
    
    @property
    def keypoints(self):
        return self._kps

    @property
    def descriptor(self):
        return self._des
