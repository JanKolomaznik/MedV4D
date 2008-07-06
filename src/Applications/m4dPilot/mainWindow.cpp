#include "mainWindow.h"

// DICOM namespace:
using namespace M4D::Dicom;

using namespace std;


mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME )
{
  Q_INIT_RESOURCE( mainWindow ); 

  QCoreApplication::setOrganizationName( ORGANIZATION_NAME );
  QCoreApplication::setApplicationName( APPLICATION_NAME );
}


// reimplemented view method
void mainWindow::view ( DcmProvider::DicomObjSet *dicomObjSet )
{
  vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->imageDataToRenderWindow( DcmProvider::DicomObjSetPtr( dicomObjSet ) ) );
}

