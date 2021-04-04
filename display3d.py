import sys
import os

import vtk


def importOBJModel(file, texturepath=None):
    importer = vtk.vtkOBJImporter()
    importer.SetFileName(file)
    if texturepath:
        path, _ = os.path.split(texturepath)
        importer.SetFileNameMTL(texturepath)
        importer.SetTexturePath(path)

    return importer

def makelight(position, focalpoint):
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(*position)
    light.SetPositional(True)
    light.SetConeAngle(10)
    light.SetFocalPoint(*focalpoint)
    light.SetDiffuseColor(1, 0, 0)
    light.SetAmbientColor(0, 1, 0)
    light.SetSpecularColor(0, 0, 1)

    return light


class Display3D(object):

    def __init__(self, file, texturefile=None):
        importer = importOBJModel(file, texturefile)
        importer.Read()

        self.renderer = importer.GetRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.4)

        """
        light = makelight((0, 0, 500), (0, 0, 50))
        lightActor = vtk.vtkLightActor()
        lightActor.SetLight(light)
        self.renderer.AddViewProp(lightActor)
        """

        self.iren = vtk.vtkRenderWindowInteractor()
        self.renwin = importer.GetRenderWindow()
        self.renwin.AddRenderer(self.renderer)
        self.iren.SetRenderWindow(self.renwin)
        self.renwin.Render()

    def render(self):
        self.iren.Start()

    def takeScreenshot(self, out='screenshot.png'):
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetScale(5)
        w2if.SetInput(self.renwin)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(out)
        writer.SetInputData(w2if.GetOutput())
        writer.Write()

if __name__ == '__main__':

    display3d = Display3D('./face_models/sonny.obj', './face_models/sonny.mtl')
    display3d.takeScreenshot()
    display3d.render()
