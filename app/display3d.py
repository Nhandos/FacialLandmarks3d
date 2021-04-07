import sys
import os

import vtk

import vtk.qt  #  see https://gitlab.kitware.com/vtk/vtk/merge_requests/1097/diffs?commit_id=1da26e338a33778a57078531256faf434fc4b593
vtk.qt.QVTKRWIBase = "QGLWidget"

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def importOBJModel(file, texturepath=None):
    importer = vtk.vtkOBJImporter()
    importer.SetFileName(file)
    if texturepath:
        path, _ = os.path.split(texturepath)
        importer.SetFileNameMTL(texturepath)
        importer.SetTexturePath(path)

    ren = vtk.vtkRenderer()
    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(ren)
    importer.SetRenderWindow(renwin)
    importer.Update()

    return ren 


class FaceModelWidget(QtWidgets.QFrame):
    

    def __init__(self, parent=None):

        super().__init__(parent)

        self._layout = QtWidgets.QVBoxLayout()
        self._vtkWidget = QVTKRenderWindowInteractor(self)
        self._layout.addWidget(self._vtkWidget)
        self.setLayout(self._layout)
        self.iren = self._vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()

    def addRendererAndUpdate(self, renderer):
        self._vtkWidget.GetRenderWindow().AddRenderer(renderer)
        self.updateRenderWindow()

    def updateRenderWindow(self):
        self._vtkWidget.GetRenderWindow().Render()


class Display3D(QtWidgets.QMainWindow):

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

        self._centralWidget = QtWidgets.QWidget(self)  # central widget house child widgets

        # FaceModelWidget
        self._faceModelWidget = FaceModelWidget(parent=self._centralWidget)  
        self._faceModelWidget.setGeometry(QtCore.QRect(280, 10, self.width, self.height))
        self._faceModelWidget.setMinimumSize(QtCore.QSize(self.width, self.height))
        self._faceModelWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._faceModelWidget.setFrameShadow(QtWidgets.QFrame.Raised)
        target = ('./face_models/sonny/sonny.obj', './face_models/sonny/sonny.mtl')  # TODO: don't hardcode
        renderer = importOBJModel(*target)
        renderer.SetBackground(.3, .6, .3)
        self._faceModelWidget.addRendererAndUpdate(renderer)

    def _initUI(self):
        pass

    def _initLayout(self):

        # Create Layout
        self._layout = QtWidgets.QHBoxLayout()

        # Define layout (order is important)
        self._layout.addWidget(self._faceModelWidget, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Final
        self._centralWidget.setLayout(self._layout)
        self.setCentralWidget(self._centralWidget)


def main():
    app = QApplication(sys.argv)
    window = Display3D()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

