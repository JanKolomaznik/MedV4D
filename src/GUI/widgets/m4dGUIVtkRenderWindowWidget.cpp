#include <QtGui/QtGui>

#include "m4dGUIVtkRenderWindowWidget.h"


m4dGUIVtkRenderWindowWidget::m4dGUIVtkRenderWindowWidget ( QVTKWidget *parent )
  : QVTKWidget( parent )
{}


void m4dGUIVtkRenderWindowWidget::addRenderer ( vtkRenderer *renderer )
{ 
  vtkRenderWindow *rWin;
  rWin = GetRenderWindow();
  rWin->AddRenderer( renderer );

  vtkRenderWindowInteractor *iren;
  iren = GetInteractor();
  iren->SetRenderWindow( rWin );
}


void m4dGUIVtkRenderWindowWidget::removeFirstRenderer ()
{ 
  vtkRenderWindow *rWin;
  rWin = GetRenderWindow();
  rWin->RemoveRenderer( rWin->GetRenderers()->GetFirstRenderer() );
}
