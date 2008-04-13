#ifndef RENDER_WINDOW_UTILS_H
#define RENDER_WINDOW_UTILS_H

#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkDICOMImageReader.h"
#include "vtkVolume.h"
#include "vtkVolumeRayCastMapper.h"
#include "vtkVolumeRayCastCompositeFunction.h"
#include "vtkImageCast.h"
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkVolumeProperty.h"


vtkRenderer *sphereToRenderWindow ();

vtkRenderer *dicomToRenderWindow ( const char *dirName );

#endif // RENDER_WINDOW_UTILS_H