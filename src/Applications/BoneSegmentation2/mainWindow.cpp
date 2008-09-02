#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

using namespace std;
using namespace M4D::Imaging;


mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME ), _inConnection( NULL ), _outConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();

	// tell mainWindow about possible connections - can be during the creation of pipeline (connections)

	// M4D::Imaging::ConnectionInterface *conn;
	// addSource( conn, "Bone segmentation", "Stage #1" );
	// addSource( conn, "Bone segmentation", "Result" );

	// add your own settings widgets
	_settings = new SettingsBox( _filter, this );

	addDockWindow( "Bone Segmentation", _settings );

	QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}


void 
mainWindow::process ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet )
{
	AbstractImage::AImagePtr inputImage = ImageFactory::CreateImageFromDICOM( dicomObjSet );

	try {
		_inConnection->PutImage( inputImage );

		_convertor->Execute();

		mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}

}

void
mainWindow::CreatePipeline()
{
	_convertor = new InImageConvertor();

	_filter = new Thresholding();
	Median2D *tmpFilter = new Median2D();
	tmpFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
		
	tmpFilter->SetRadius( 4 );

	_pipeline.AddFilter( _convertor );
	_pipeline.AddFilter( _filter );
	_pipeline.AddFilter( tmpFilter );
	;

	_inConnection = dynamic_cast<AbstractImageConnectionInterface*>( &_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 );
	_tmpConnection = dynamic_cast<AbstractImageConnectionInterface*>( &_pipeline.MakeConnection( *_filter, 0, *tmpFilter, 0 ) );
	_outConnection = dynamic_cast<AbstractImageConnectionInterface*>( &_pipeline.MakeOutputConnection( *tmpFilter, 0, true ) );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Bone segmentation", "Input" );
	addSource( _tmpConnection, "Bone segmentation", "Stage #1" );
	addSource( _outConnection, "Bone segmentation", "Result" );

	_notifier =  new Notifier( this );
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}

