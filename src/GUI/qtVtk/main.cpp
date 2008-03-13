#include <QtGui/QtGui>

#include "uiQtVtkMainWindow.h"

#include "../../include/M4DCommon.h"

// not a final solution
#ifdef OS_WIN
#include "projectConfig.h"
#endif

#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h"


int main ( int argc, char *argv[] )
{
  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  QMainWindow *form = new QMainWindow;
  Ui::MainWindow ui;
  ui.setupUi( form );


  // create sphere geometry
  vtkSphereSource *sphere = vtkSphereSource::New();
  sphere->SetRadius( 1.0 );
  sphere->SetThetaResolution( 8 );
  sphere->SetPhiResolution( 8 );

  // map to graphics library
  vtkPolyDataMapper *map = vtkPolyDataMapper::New();
  map->SetInput( sphere->GetOutput() );

  // actor coordinates geometry, properties, transformation
  vtkActor *aSphere = vtkActor::New();
  aSphere->SetMapper( map );
  aSphere->GetProperty()->SetColor( 1, 0, 0 );	// sphere color blue

  // a renderer and render window
  vtkRenderer *ren1 = vtkRenderer::New();
  vtkRenderWindow *renWin = ui.qvtkWidget->GetRenderWindow();
  renWin->AddRenderer( ren1 );					// adding first renderer to renderWindow

  // an interactor
  vtkRenderWindowInteractor *iren = ui.qvtkWidget->GetInteractor();
  iren->SetRenderWindow( renWin );

  // add the actor to the scene
  ren1->AddActor( aSphere );
  ren1->SetBackground( 1, 1, 1 );				// background color white

  // render an image (lights and cameras are created automatically)
  // renWin->Render();

  // begin mouse interaction
  // iren->Start();

  
  form->show();

  return app.exec();
}