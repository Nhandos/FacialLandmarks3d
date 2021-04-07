import sys
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon

from mywidgets import FaceModelWidget 
from helpers import importOBJModel


class MainToolBar(QtWidgets.QToolBar):
    
    screenshotTriggered = QtCore.pyqtSignal()
    loadLandmarksTriggered = QtCore.pyqtSignal()
    detectKPTriggered = QtCore.pyqtSignal()
   

    def __init__(self, title, parent=None):
        super().__init__(title, parent=parent)
        self.initToolBarActions()
        self.initToolBarActionsLayout()

    def initToolBarActions(self):
        # Exit
        self.exitAct = QtWidgets.QAction(QIcon('./res/door_out.png'), 'Exit', self)
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.triggered.connect(QtWidgets.qApp.quit)

        # Screenshot
        self.screenshotAct = QtWidgets.QAction(QIcon('./res/screenshot.png'), 'Screenshot', self)
        self.screenshotAct.setShortcut('Ctrl+J')
        self.screenshotAct.triggered.connect(self.screenshotTriggered)

        # LoadLandMarks
        self.loadlandmarksAct = QtWidgets.QAction(QIcon('./res/landmarks.png'), 'Load Landmarks', self)
        self.loadlandmarksAct.triggered.connect(self.loadLandmarksTriggered)

        # Detect SIFT
        self.detectKPAct = QtWidgets.QAction(QIcon('./res/keypoints.png'), 'Keypoints', self)
        self.detectKPAct.triggered.connect(self.detectKPTriggered)

    def initToolBarActionsLayout(self):
        self.addAction(self.exitAct)
        self.addAction(self.screenshotAct) 
        self.addAction(self.loadlandmarksAct)
        self.addAction(self.detectKPAct)

        
class MainApp(QtWidgets.QMainWindow):

    def __init__(self, width=640, height=640, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.width, self.height = width, height
        self._initialise()
        self.show()

    def _initialise(self):
        self._initWidgets()
        self._initUI()
        self._initLayout()

    def _initWidgets(self):

        self._centralWidget = FaceModelWidget(parent=self)
        self._centralWidget.setGeometry(QtCore.QRect(280, 10, self.width, self.height))
        self._centralWidget.setMinimumSize(QtCore.QSize(self.width, self.height))
        self._centralWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._centralWidget.setFrameShadow(QtWidgets.QFrame.Raised)
        target = ('../../face_models/sonny/sonny.obj', 
                  '../../face_models/sonny/sonny.mtl')  # TODO: don't hardcode
        renderer = importOBJModel(*target)
        renderer.SetBackground(.3, .6, .3)
        self._centralWidget.addRendererAndUpdate(renderer)

        self._toolbar = MainToolBar('MainToolBar')
        self.addToolBar(self._toolbar)


    def _initUI(self):
        self._toolbar.screenshotTriggered.connect(self._takeScreenshot)
        self._toolbar.detectKPTriggered.connect(self._centralWidget.detectKP)

    def _initLayout(self):

        # Create Layout
        self._layout = QtWidgets.QHBoxLayout()

        # Define layout (order is important)
        self._layout.addWidget(self._centralWidget, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Final
        self.setCentralWidget(self._centralWidget)

    def _takeScreenshot(self):

        # Open file dialog
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, 
                "QFileDialg.getSaveFileName()", "", 
                "All Files (*);;JPEG files (*.jpeg *.jpg, *.jpe);;Portable Network Graphics (*.png)", 
                options=options)
        if fileName:
            img = self._centralWidget.takeScreenshot()
            cv2.imwrite(filename, img)

    def _loadLandmarks(self):

        # Open file dialog
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, 
                "QFileDialg.getSaveFileName()", "", 
                "All Files (*);;JPEG files (*.jpeg *.jpg, *.jpe);;Portable Network Graphics (*.png)", 
                options=options)

        if fileName:
            img = self._centralWidget.takeScreenshot()
            cv2.imwrite(filename, img)
        
        landmarks = ImportLandmarks(



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

