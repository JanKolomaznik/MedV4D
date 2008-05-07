#ifndef M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H
#define M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H

#include <QWidget>
#include "QVTKWidget.h"

// VTK includes
#include "vtkRenderer.h"


class m4dGUIVtkRenderWindowWidget: public QVTKWidget
{
  Q_OBJECT

  public:
    m4dGUIVtkRenderWindowWidget ( QVTKWidget *parent = 0 );

    void addRenderer ( vtkRenderer *ren );
    void removeFirstRenderer ();

    vtkRenderer *imageDataToRenderWindow ();

    // just for testing purposes - won't be in final ver.
    vtkRenderer *sphereToRenderWindow ();
    vtkRenderer *dicomToRenderWindow ( const char *dirName );
};

#endif // M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H