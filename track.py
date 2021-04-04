import argparse

import cv2

from display import Display2D
from keypoints import Frame, FrameMatcher

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='Video file')
parser.add_argument('--headless', action='store_true',
    help='no display')
parser.add_argument('--save', type=str,
    help='output file')

def main(args):

    capture = cv2.VideoCapture(args.video)

    W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(capture.get(cv2.CAP_PROP_FPS))

    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (W, H))

    if W > 1024:
        downscale = 1024.0 / W
        H = int(H * downscale)
        W = 1024

    if not args.headless:
        display2d = Display2D(W, H)

    ret, frame = capture.read()
    if not ret:
        print("Fatal - failed to load video")
        exit(-1)

    matcher = FrameMatcher()
    prevframe = Frame(frame)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("No more frames to read")
            break

        frame = Frame(frame)
        if frame.roi is not None and prevframe.roi is not None:
            matches = matcher(frame, prevframe)
            print('matches:' ,len(matches))
        else:
            print('no ROI')

        if not args.headless:
            display2d.paint(cv2.resize(frame.getAnnotated(), (W, H)))
            
        if args.save:
            writer.write(frame.getAnnotated())
            
        prevframe = frame


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
