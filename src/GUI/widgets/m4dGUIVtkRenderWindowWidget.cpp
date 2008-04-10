#include <QtGui/QtGui>

#include "m4dGUIVtkRenderWindowWidget.h"


m4dGUIVtkRenderWindowWidget::m4dGUIVtkRenderWindowWidget ( QWidget *parent )
  : QWidget( parent )
{
  renWin = new QVTKWidget;
}