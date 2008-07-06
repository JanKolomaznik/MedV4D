#include "mainWindow.h"

// DICOM namespace:
using namespace M4D::Dicom;

using namespace std;


mainWindow::mainWindow ()
{}


void mainWindow::view ( DcmProvider::DicomObjSet *dicomObjSet )
{
  vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->imageDataToRenderWindow( DcmProvider::DicomObjSetPtr( dicomObjSet ) ) );
}

