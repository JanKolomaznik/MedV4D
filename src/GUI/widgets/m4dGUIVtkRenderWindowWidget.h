#ifndef M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H
#define M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H

#include <QtGui/QWidget>
#include "QVTKWidget.h"


// VTK includes
#include "../m4dPilot/renderWindowUtils.h"

class m4dGUIVtkRenderWindowWidget: public QVTKWidget
{
  Q_OBJECT

  public:
    m4dGUIVtkRenderWindowWidget ( QVTKWidget *parent = 0 );

    void addRenderer ( vtkRenderer *ren );
    void removeFirstRenderer ();
};

#endif // M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H