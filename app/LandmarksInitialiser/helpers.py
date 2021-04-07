import os
import vtk

def importOBJModel(modelfile, texturepath=None):
    importer = vtk.vtkOBJImporter()
    importer.SetFileName(modelfile)
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

