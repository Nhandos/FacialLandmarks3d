import argparse

import cv2
from matplotlib import pyplot as plt

from keypoints import FeatureExtractor, FeatureMatcher, FaceDetector


parser = argparse.ArgumentParser(description='Keypoints matcher')
parser.add_argument('image1', type=str)
parser.add_argument('image2', type=str)

def roiToRect(roi):
    x,y,w,h = roi
    return ((x,y),(x+w,y+h))

def main(args):

    facedetector = FaceDetector()
    extractor = FeatureExtractor()
    matcher = FeatureMatcher()

    img1 = cv2.imread(args.image1)
    roi1 = facedetector(img1)
    img2 = cv2.imread(args.image2)
    roi2 = facedetector(img2)

    kp1, des1, _ = extractor(img1, roi1)
    kp2, des2, _ = extractor(img2, roi2)
    matches = matcher(kp1, des1, kp2, des2, loweratio=0.8)

    kpimg1 = cv2.drawKeypoints(img1, kp1, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpimg1 = cv2.rectangle(kpimg1,*roiToRect(roi1),(0,255,0),3)
    kpimg2 = cv2.drawKeypoints(img2, kp2, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpimg2 = cv2.rectangle(kpimg2,*roiToRect(roi2),(0,255,0),3)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure()
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(kpimg1, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(kpimg2, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    main(parser.parse_args())

