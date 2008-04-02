#include <QtGui/QtGui>

#include "qtVtkMainWindow.h"


MainWindow::MainWindow( QMainWindow *parent )
    : QMainWindow( parent )
{
  ui.setupUi( this );

  // just for testing:
  renSphere = sphereToRenderWindow();
  renDicom  = dicomToRenderWindow();

  renWin = ui.qvtkWidget->GetRenderWindow();
  if ( renSphere != NULL ) {
    renWin->AddRenderer( renSphere );
  }
  
  iren = ui.qvtkWidget->GetInteractor();
  iren->SetRenderWindow( renWin );
}

void MainWindow::on_SphereRadio_clicked ()
{
  if ( renSphere != NULL ) {
    renWin->RemoveRenderer( renDicom );
    renWin->AddRenderer( renSphere );
  }
}

void MainWindow::on_DicomRadio_clicked ()
{
  if ( renDicom != NULL ) {
    renWin->RemoveRenderer( renSphere );
    renWin->AddRenderer( renDicom );
  }
}