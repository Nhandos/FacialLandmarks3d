import os

import cv2
import numpy as np
import pygame
from PyQt5 import QtCore, QtGui

from pygame.locals import RESIZABLE


class Display2D(object):

    
    def __init__(self, W, H):

        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def paint(self, image: np.ndarray):
        # junk
        for _ in pygame.event.get():
            pass

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pygame.surfarray.blit_array(self.surface, image.swapaxes(0, 1)[:, :, [0,1,2]])
        self.screen.blit(self.surface, (0,0))

        pygame.display.flip()


class Display3D(QtGui.QWindow):

    def __init__(self):
        pass


if __name__ == '__main__':
    
   # capture = cv2.VideoCapture(os.path.basename('videos/test_countryroad.mp4'))
    capture = cv2.VideoCapture('./videos/test_countryroad.mp4')

    W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if W > 1024:
        downscale = 1024.0 / W
        H = int(H * downscale)
        W = 1024

    display2d = Display2D(W, H)

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (W, H))
        display2d.paint(frame)