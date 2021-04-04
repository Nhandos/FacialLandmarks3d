from typing import Tuple

import cv2
import numpy as np


class FaceDetector(object):
    

    def __init__(self):
        self.face_cascade = \
            cv2.CascadeClassifier('./res/haarcascade_frontalface_alt.xml')

    def __call__(self, image:np.ndarray):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None

        return faces[-1]  # todo: return the largest ROI face


class FeatureExtractor(object):

    def __init__(self, detector='SIFT'):

        if detector == 'ORB':
            self.detector = cv2.ORB_create()
        elif detector == 'SIFT':
            self.detector = cv2.SIFT_create(
                nfeatures=0,
                nOctaveLayers=3,
                contrastThreshold=0.05,  # 
                edgeThreshold=100
            )

    def __call__(self, image: np.ndarray, roi: Tuple[int, int, int, int]=None):
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if roi is not None:
            # mask roi
            x, y, w, h = roi
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[y:y+h, x:x+w, ...] = gray[y:y+h, x:x+w, ...]

            # image processing
            #mask = cv2.equalizeHist(mask)
            #mask = np.clip(mask * 1.2, 0.0, 255.0).astype(np.uint8)
        else:
            mask = gray

        # features detection + extraction 
        pts = cv2.goodFeaturesToTrack(mask, 3000,  
            qualityLevel=0.00001, minDistance=7, mask=mask)
        
        #kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        #kps, des = self.detector.compute(image, kps)
        kps, des = self.detector.detectAndCompute(mask, None)

        return kps, des, mask


class FeatureMatcher(object):
    
    def __init__(self, normtype=cv2.NORM_L2):
        self.bfmatcher = cv2.BFMatcher(normtype, crossCheck=False)

    def __call__(self, kp1, des1, kp2, des2, loweratio=0.7):

        # match kps
        matches = self.bfmatcher.knnMatch(des1, des2, 2) # requires atleast 2 nearest matches

        # loweratio filter
        good = []
        for i, knnmatch in enumerate(matches):
            m, n = knnmatch[:2]
            if m.distance < loweratio * n.distance:
                good.append([m])

        return good


class FrameMatcher(FeatureMatcher):

    def __init__(self, normtype=cv2.NORM_L2):
        self.bfmatcher = cv2.BFMatcher(normtype, crossCheck=False)

    def __call__(self, frame1, frame2, loweratio=0.7):
        kp1, des1 = frame1.keypoints, frame1.descriptor
        kp2, des2 = frame2.keypoints, frame2.descriptor

        return super().__call__(kp1, des1, kp2, des2, loweratio=0.7)



class Frame(object):

    facedetector = FaceDetector()
    featureextractor = FeatureExtractor()

    def __init__(self, image: np.ndarray):

        self.image = image 

        if self.image is not None:
            self.roi = self.facedetector(self.image)
            self._kps, self._des, self._mask = \
                self.featureextractor(self.image, self.roi)

    def getAnnotated(self, matches=None):

        ret = self.image
        # ROI
        if self.roi is not None:
            x, y, w, h = self.roi
            ret = cv2.rectangle(ret, (x, y), (x + w, y + h), (255, 0, 0), 2)
            ret[y:y+h,x:x+h] = np.repeat(self._mask[y:y+h,x:x+h], 3).reshape(w, h, 3)

        # keypoints labelling
        ret = cv2.drawKeypoints(ret, self.keypoints, None, color=(0, 0, 255), 
            flags=0)

        return ret
    
    @property
    def keypoints(self):
        return self._kps

    @property
    def descriptor(self):
        return self._des
