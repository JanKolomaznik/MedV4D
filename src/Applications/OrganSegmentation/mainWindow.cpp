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
	/*QObject::connect( _settings, SIGNAL( SetSegmentationSignal( uint32 )),
		this, SLOT(SetSegmentationSlot( uint32 )) );*/

	addDockWindow( "Organ Segmentation", _settings, DOCKED_DOCK_WINDOW );

	/*_segmentationWidget = new SegmentationWidget();
	addDesktopWidget( _segmentationWidget );*/

	_segmentationViewerWidget = new SegmentationViewerWidget();
	addViewerDesktop( _segmentationViewerWidget );

	//Add desktop for viewing results
	addDesktopWidget( MainManager::Instance().GetResultsPage() );

	//Connect segmentation managers to mainWindow
	SegmentationManagersList::iterator it;
	for( it = segmentationManagers.begin(); it != segmentationManagers.end(); ++it ) {

		QObject::connect( *it, SIGNAL(ManagerActivated( ManagerActivationInfo )), this, SLOT( ManagerActivatedSlot( ManagerActivationInfo ) ) );
		QObject::connect( *it, SIGNAL(WantsViewerUpdate()), _segmentationViewerWidget, SLOT( UpdateViewers() ) );
		QObject::connect( *it, SIGNAL(ProcessResults( ResultsInfo )), this, SLOT( ProcessResults( ResultsInfo ) ) );
		QObject::connect( *it, SIGNAL(ErrorMessageSignal( const QString & )), this, SLOT( ErrorMessageSlot( const QString & ) ) );

	}
}


void 
mainWindow::process ( M4D::Imaging::AbstractDataSet::Ptr inputDataSet )
{
	MainManager::Instance().InitInput( inputDataSet );

	SetToDefaultSlot();
}

void
mainWindow::ManagerActivatedSlot( ManagerActivationInfo info )
{
	_segmentationViewerWidget->Activate( info.connection, info.specState );

	_settings->setCurrentWidget( info.panel );

	switchToViewerDesktop( 2, false );
}

void
mainWindow::ProcessResults( ResultsInfo info )
{
	MainManager::Instance().ProcessResultDatasets( info.image, info.geometry );

	switchToDesktopWidget();
	_settings->SetToResults();
}

void
mainWindow::SetToDefaultSlot()
{
	_settings->SetToDefault();
	switchToViewerDesktop( 1, true );
}

/*void
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
*/



