from enum import Enum


class Landmark(object):

    DEFAULT_LANDMARK_RADIUS = 5

    def __init__(self, plane, name, position):

        self.group = group
        self.name = name
        self.position = position 

    def makeVTKActor(self):

        source = vtk.vtkSphereSource()
        source.SetCenter(*self.position)
        source.SetRadius(self.DEFAULT_LANDMARK_RADIUS)

        mapper = vkt.vtkPolyDataMapper()
        mapper.SetInput(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

