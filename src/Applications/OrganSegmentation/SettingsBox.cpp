#include "SettingsBox.h"
#include "mainWindow.h"
#include "SegmentationManager.h"
#include "ManualSegmentationManager.h"
#include "KidneySegmentationManager.h"


SettingsBox
::SettingsBox( QWidget * parent )
	: _parent( parent )
{
	setMinimumWidth( MINIMUM_WIDTH );

	InitializeSementationGUI();	
}

void
SettingsBox
::InitializeSementationGUI()
{
	_mainSettings = new QWidget;
	QVBoxLayout *layout = new QVBoxLayout;

	_mainSettings->setLayout(layout);
	addWidget( _mainSettings );

	SegmentationManagersList::iterator it;
	for( it = segmentationManagers.begin(); it != segmentationManagers.end(); ++it ) {
		//Add activating button for segmentation manager
		QPushButton *button = new QPushButton( tr( (*it)->GetName().data() ) );
		layout->addWidget( button );
		QObject::connect( button, SIGNAL(clicked()), *it, SLOT( ActivateManager() ) );

		//Add manager's GUI
		addWidget( (*it)->GetGUI() );
	}

	layout->addStretch( 5 );
}

void
SettingsBox
::SetToDefault()
{
	//emit SetSegmentationSignal( stDefault );
	setCurrentWidget( _mainSettings );
}


/*void
SettingsBox
::SetToManualSegmentation()
{
	ManualSegmentationManager::Instance().Activate( MainManager::Instance().GetInputImage() );
	
	emit SetSegmentationSignal( stMANUAL );
	setCurrentWidget( _manualSegmSettings );
}

void
SettingsBox
::SetToKidneySegmentation()
{
	KidneySegmentationManager::Instance().Activate( MainManager::Instance().GetInputImage() );
	
	emit SetSegmentationSignal( stKIDNEYS );
	setCurrentWidget( _kidneySegmSettings );
}

void
SettingsBox
::BeginManualCorrection()
{
	ManualSegmentationManager::Instance().Activate( 
			KidneySegmentationManager::Instance().GetInputImage(), 
			KidneySegmentationManager::Instance().GetOutputGeometry() 
			);
	
	emit SetSegmentationSignal( stMANUAL );
	setCurrentWidget( _manualSegmSettings );
}

void
SettingsBox
::PolesSetKidneySegm()
{
	KidneySegmentationManager::Instance().PolesSet();
}

void
SettingsBox
::StartSegmentationKidneySegm()
{
	KidneySegmentationManager::Instance().StartSegmentation();

}

void
SettingsBox
::ProcessResultsManual()
{
	MainManager::Instance().ProcessResultDatasets(
			ManualSegmentationManager::Instance().GetInputImage(),
			ManualSegmentationManager::Instance().GetOutputGeometry()
			);
}*/
