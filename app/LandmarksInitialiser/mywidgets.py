import vtk.qt  
vtk.qt.QVTKRWIBase = "QGLWidget"  # prevents improper rendering
import cv2
import numpy as np
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.numpy_support import vtk_to_numpy
from PyQt5 import QtCore, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


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

    def takeScreenshot(self, magnification=1):
        # Takes screenshot using current active camera
        
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(self._vtkWidget.GetRenderWindow())
        windowToImageFilter.SetScale(magnification)
        windowToImageFilter.Update()
        image = windowToImageFilter.GetOutput()

        cols, rows, _ = image.GetDimensions()
        vtkArr = image.GetPointData().GetScalars()
        components = vtkArr.GetNumberOfComponents()       

        img = vtk_to_numpy(vtkArr).reshape(rows, cols, components)
        img = np.flip(img, axis=0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def detectKP(self):
        pass


