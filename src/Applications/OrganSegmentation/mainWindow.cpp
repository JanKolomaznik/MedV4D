#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

#include "MainManager.h"

using namespace std;
using namespace M4D::Imaging;


mainWindow::mainWindow ()
  : m4dGUIMainWindow2( APPLICATION_NAME, ORGANIZATION_NAME )//, _inConnection( NULL ), _outConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 

//	CreatePipeline();

	// tell mainWindow about possible connections - can be during the creation of pipeline (connections)

	// M4D::Imaging::ConnectionInterface *conn;
	// addSource( conn, "Bone segmentation", "Stage #1" );
	// addSource( conn, "Bone segmentation", "Result" );

	//addSource( MainManager::GetInputConnection(), "Input", "Input" );

	mainViewerDesktop->setDefaultConnection( MainManager::GetInputConnection() );
	//mainViewerDesktop->setConnectionForAll( mainViewerDesktop->getDefaultConnection() );

	// add your own settings widgets
	_settings = new SettingsBox( /*_filter, */this );

	addDockWindow( "Organ Segmentation", _settings );

	_manualSegmentation = new ManualSegmentationWidget();
	addDesktopWidget( _manualSegmentation );

	//QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}


void 
mainWindow::process ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet )
{
	MainManager::InitInput( dicomObjSet );

	mainViewerDesktop->setConnectionForAll( mainViewerDesktop->getDefaultConnection() );

	/*AbstractImage::AImagePtr inputImage = ImageFactory::CreateImageFromDICOM( dicomObjSet );

	try {
		_inConnection->PutImage( inputImage );

		_convertor->Execute();

		mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}*/

}

