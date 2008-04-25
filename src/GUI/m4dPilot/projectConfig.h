#ifndef PROJECT_CONFIG_H
#define PROJECT_CONFIG_H

#ifdef _DEBUG
  #pragma comment(lib,"vtkCommond.lib")
  #pragma comment(lib,"vtkDICOMParserd.lib")
  #pragma comment(lib,"vtkFilteringd.lib")
  #pragma comment(lib,"vtkGenericFilteringd.lib")
  #pragma comment(lib,"vtkGraphicsd.lib")
  #pragma comment(lib,"vtkHybridd.lib")
  #pragma comment(lib,"vtkIOd.lib")
  #pragma comment(lib,"vtkImagingd.lib")
  #pragma comment(lib,"vtkNetCDFd.lib")
  #pragma comment(lib,"vtkRenderingd.lib")
  #pragma comment(lib,"vtkVolumeRenderingd.lib")
  #pragma comment(lib,"vtkWidgetsd.lib")
  #pragma comment(lib,"vtkexoIIcd.lib")
  #pragma comment(lib,"vtkexpatd.lib")
  #pragma comment(lib,"vtkfreetyped.lib")
  #pragma comment(lib,"vtkftgld.lib")
  #pragma comment(lib,"vtkjpegd.lib")
  #pragma comment(lib,"vtkpngd.lib")
  #pragma comment(lib,"vtksysd.lib")
  #pragma comment(lib,"vtktiffd.lib")
  #pragma comment(lib,"vtkzlibd.lib")
  #pragma comment(lib,"QVTKd.lib")

#else // _RELEASE
  #pragma comment(lib,"vtkCommon.lib")
  #pragma comment(lib,"vtkDICOMParser.lib")
  #pragma comment(lib,"vtkFiltering.lib")
  #pragma comment(lib,"vtkGenericFiltering.lib")
  #pragma comment(lib,"vtkGraphics.lib")
  #pragma comment(lib,"vtkHybrid.lib")
  #pragma comment(lib,"vtkIO.lib")
  #pragma comment(lib,"vtkImaging.lib")
  #pragma comment(lib,"vtkNetCDF.lib")
  #pragma comment(lib,"vtkRendering.lib")
  #pragma comment(lib,"vtkVolumeRendering.lib")
  #pragma comment(lib,"vtkWidgets.lib")
  #pragma comment(lib,"vtkexoIIc.lib")
  #pragma comment(lib,"vtkexpat.lib")
  #pragma comment(lib,"vtkfreetype.lib")
  #pragma comment(lib,"vtkftgl.lib")
  #pragma comment(lib,"vtkjpeg.lib")
  #pragma comment(lib,"vtkpng.lib")
  #pragma comment(lib,"vtksys.lib")
  #pragma comment(lib,"vtktiff.lib")
  #pragma comment(lib,"vtkzlib.lib")
  #pragma comment(lib,"QVTK.lib")

#endif


#endif // PROJECT_CONFIG_H