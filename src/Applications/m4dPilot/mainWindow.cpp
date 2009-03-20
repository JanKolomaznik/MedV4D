#include "mainWindow.h"

using namespace std;
using namespace M4D::Imaging;

mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME )
{
  Q_INIT_RESOURCE( mainWindow ); 

  // createPipeline();

  // tell mainWindow about possible connections - can be during the creation of pipeline (connections)

  // M4D::Imaging::ConnectionInterface *conn;
  // addSource( conn, "Bone segmentation", "Stage #1" );
  // addSource( conn, "Bone segmentation", "Result" );
  
  // add your own settings widgets

  // addDockWindow( "Bone Segmentation", new QListWidget );
}


void mainWindow::process ( M4D::Dicom::DicomObjSetPtr dicomObjSet )
{
	AbstractImage::Ptr inputImage = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );

	try {

    ConnectionTyped<AbstractImage> *conn = new ConnectionTyped<AbstractImage>();
		conn->PutDataset( inputImage );

		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		conn->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

	} 
	catch ( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}
