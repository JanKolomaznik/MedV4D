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


	mainViewerDesktop->setDefaultConnection( MainManager::GetInputConnection().get() );
	//mainViewerDesktop->setConnectionForAll( mainViewerDesktop->getDefaultConnection() );

	// add your own settings widgets
	_settings = new SettingsBox( this );
	QObject::connect( _settings, SIGNAL(SetToManualSignal()),
		this, SLOT(SetToManual()) );

	addDockWindow( "Organ Segmentation", _settings );

	_manualSegmentation = new ManualSegmentationWidget();
	addDesktopWidget( _manualSegmentation );
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

void
mainWindow::SetToManual()
{
	_manualSegmentation->Activate();

	stackWidget->setCurrentWidget( _manualSegmentation );
}


