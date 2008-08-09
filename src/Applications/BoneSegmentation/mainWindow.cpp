#include "mainWindow.h"
#include "Imaging/filters/ThresholdingFilter.h"

using namespace std;
using namespace M4D::Imaging;

typedef Image< uint16, 3 > ImageType;
typedef ThresholdingFilter< ImageType > Thresholding;
typedef ImageConnection< ImageType > InConnection;

mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME )
{
	Q_INIT_RESOURCE( mainWindow ); 

	QCoreApplication::setOrganizationName( ORGANIZATION_NAME );
	QCoreApplication::setApplicationName( APPLICATION_NAME );

	CreatePipeline();

	// tell mainWindow about possible connections - can be during the creation of pipeline (connections)

	// M4D::Imaging::ConnectionInterface *conn;
	// addSource( conn, "Bone segmentation", "Stage #1" );
	// addSource( conn, "Bone segmentation", "Result" );

	// add your own settings widgets

	addDockWindow( "Bone Segmentation", new QListWidget );
}

void 
mainWindow::process ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet )
{
	AbstractImage::AImagePtr inputImage = ImageFactory::CreateImageFromDICOM( dicomObjSet );

	unsigned dim = inputImage->GetDimension(); 
	int type     = inputImage->GetElementTypeID();

	if ( dim != 3 || type != NTID_UNSIGNED_SHORT ) {
		//TODO throw exception

		QMessageBox::critical( this, tr( "Exception" ), tr( "Bad type" ) );
		return;
	}
	try {
		_inConnection->PutImage( inputImage );

		//mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		//conn->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}

}

void
mainWindow::CreatePipeline()
{
	Thresholding *filter = new Thresholding();

	_pipeline.AddFilter( filter );
	_inConnection = dynamic_cast<AbstractImageConnection*>( &_pipeline.MakeInputConnection( *filter, 0, false ) );
	_outConnection = dynamic_cast<AbstractImageConnection*>( &_pipeline.MakeOutputConnection( *filter, 0, true ) );

	addSource( _inConnection, "Bone segmentation", "Stage #1" );
	addSource( _outConnection, "Bone segmentation", "Result" );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}
}

