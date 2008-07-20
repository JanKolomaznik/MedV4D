#ifndef M4D_GUI_VTK_VIEWER_WIDGET_H
#define M4D_GUI_VTK_VIEWER_WIDGET_H

#include <QWidget>
#include "QVTKWidget.h"

// VTK includes
#include "vtkRenderer.h"

// just for imageDataToRenderWindow ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects )
// DICOM includes:
#include "Common.h"
#include "dicomConn/DICOMServiceProvider.h"


class m4dGUIVtkViewerWidget: public QVTKWidget
{
  Q_OBJECT

  public:
    m4dGUIVtkViewerWidget ( QVTKWidget *parent = 0 );

    void addRenderer ( vtkRenderer *ren );

    vtkRenderer *imageDataToRenderWindow ();

    // just for testing ImageFactory::CreateImageFromDICOM 
    vtkRenderer *imageDataToRenderWindow ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects );

    // just for testing purposes - won't be in final ver.
    vtkRenderer *sphereToRenderWindow ();
    vtkRenderer *dicomToRenderWindow ( const char *dirName );
};

#endif // M4D_GUI_VTK_VIEWER_WIDGET_H

