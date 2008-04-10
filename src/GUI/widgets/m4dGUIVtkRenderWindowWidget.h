#ifndef M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H
#define M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H

#include <QtGui/QWidget>
#include "QVTKWidget.h"


class m4dGUIVtkRenderWindowWidget: public QWidget
{
  Q_OBJECT

  public:
    m4dGUIVtkRenderWindowWidget ( QWidget *parent = 0 );

    // addRenderer ( vtkRenderer ren );

  private:
    QVTKWidget *renWin;
};

#endif // M4D_GUI_VTK_RENDER_WINDOW_WIDGET_H