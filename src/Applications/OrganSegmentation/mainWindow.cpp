#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

#include "MainManager.h"
#include "ManualSegmentationManager.h"
#include "KidneySegmentationManager.h"

using namespace std;
using namespace M4D::Imaging;

mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME )//, _inConnection( NULL ), _outConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 


	currentViewerDesktop->setDefaultConnection( MainManager::Instance().GetInputConnection() );
	currentViewerDesktop->setConnectionForAll( currentViewerDesktop->getDefaultConnection() );

	// add your own settings widgets
	_settings = new SettingsBox( this );
	QObject::connect( _settings, SIGNAL( SetSegmentationSignal( uint32 )),
		this, SLOT(SetSegmentationSlot( uint32 )) );

	addDockWindow( "Organ Segmentation", _settings, DOCKED_DOCK_WINDOW );

	/*_segmentationWidget = new SegmentationWidget();
	addDesktopWidget( _segmentationWidget );*/

	_segmentationViewerWidget = new SegmentationViewerWidget();
	addViewerDesktop( _segmentationViewerWidget );
}


void 
mainWindow::process ( M4D::Imaging::AbstractDataSet::Ptr inputDataSet )
{
	MainManager::Instance().InitInput( inputDataSet );

	_settings->SetToDefault();

	//mainViewerDesktop->setConnectionForAll( mainViewerDesktop->getDefaultConnection() );

	/*AbstractImage::Ptr inputImage = ImageFactory::CreateImageFromDICOM( dicomObjSet );

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
mainWindow::SetSegmentationSlot( uint32 segType )
{
	switch( segType ) {
	case stMANUAL:
		_segmentationViewerWidget->Activate( ManualSegmentationManager::Instance().GetInputConnection(), ManualSegmentationManager::Instance().GetSpecialState() );

		switchToViewerDesktop( 2, false );
		break;
	case stKIDNEYS:
		_segmentationViewerWidget->Activate( KidneySegmentationManager::Instance().GetInputConnection(), KidneySegmentationManager::Instance().GetSpecialState() );

		//stackWidget->setCurrentWidget( _segmentationWidget );
		switchToViewerDesktop( 2, false );
		break;
	case stDefault:
		//stackWidget->setCurrentIndex( 0 );
		switchToViewerDesktop( 1, true );
		break;
	default:
		ASSERT( false );
		break;
	}
}


/*mainWindow::mainWindow ()
  : m4dGUIMainWindow2( APPLICATION_NAME, ORGANIZATION_NAME )//, _inConnection( NULL ), _outConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 


	currentViewerDesktop->setDefaultConnection( MainManager::Instance().GetInputConnection() );
	currentViewerDesktop->setConnectionForAll( currentViewerDesktop->getDefaultConnection() );

	// add your own settings widgets
	_settings = new SettingsBox( this );
	QObject::connect( _settings, SIGNAL( SetSegmentationSignal( uint32 )),
		this, SLOT(SetSegmentationSlot( uint32 )) );

	addDockWindow( "Organ Segmentation", _settings );

	_segmentationWidget = new SegmentationWidget();
	addDesktopWidget( _segmentationWidget );
}*/


/*void 
mainWindow::process ( M4D::Dicom::DicomObjSetPtr dicomObjSet )
{
	MainManager::Instance().InitInput( dicomObjSet );

	_settings->SetToDefault();

	
}

void
mainWindow::SetSegmentationSlot( uint32 segType )
{
	switch( segType ) {
	case stMANUAL:
		_segmentationWidget->Activate( ManualSegmentationManager::Instance().GetInputConnection(), ManualSegmentationManager::Instance().GetSpecialState() );

		stackWidget->setCurrentWidget( _segmentationWidget );
		break;
	case stKIDNEYS:
		_segmentationWidget->Activate( KidneySegmentationManager::Instance().GetInputConnection(), KidneySegmentationManager::Instance().GetSpecialState() );

		stackWidget->setCurrentWidget( _segmentationWidget );
		break;
	case stDefault:
		stackWidget->setCurrentIndex( 0 );
		break;
	default:
		ASSERT( false );
		break;
	}
}*/


