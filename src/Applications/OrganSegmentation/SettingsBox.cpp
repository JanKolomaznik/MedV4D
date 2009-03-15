#include "SettingsBox.h"
#include "mainWindow.h"
#include "ManualSegmentationManager.h"
#include "KidneySegmentationManager.h"


SettingsBox
::SettingsBox( QWidget * parent )
	: _parent( parent )
{
	CreateWidgets();	
}

void
SettingsBox
::CreateWidgets()
{
	setMinimumWidth( MINIMUM_WIDTH );

	
	//Create first page of settings
	_mainSettings = new QWidget;
	{
		QVBoxLayout *layout;
		QPushButton *button;
		layout = new QVBoxLayout;
		
		button = new QPushButton( tr( "Manual Segmentation" ) );
		layout->addWidget( button );
		QObject::connect( button, SIGNAL(clicked()), this, SLOT( SetToManualSegmentation()) );

		button = new QPushButton( tr( "Kidney Segmentation" ) );
		layout->addWidget( button );
		QObject::connect( button, SIGNAL(clicked()), this, SLOT( SetToKidneySegmentation()) );

		layout->addStretch( 5 );
		_mainSettings->setLayout(layout);
	}
	addWidget( _mainSettings );

	_manualSegmSettings = new QWidget;
	{
		/*QVBoxLayout *layout;
		QPushButton *button;
		layout = new QVBoxLayout;
		button = new QPushButton( tr( "Finished" ) );
			layout->addWidget( button );

		layout->addStretch( 5 );

		_manualSegmSettings->setLayout(layout);*/
		CreateManualGUI();
	}
	addWidget( _manualSegmSettings );

	_kidneySegmSettings = new QWidget;
	{
		QVBoxLayout *layout;
		QPushButton *button;
		layout = new QVBoxLayout;
		button = new QPushButton( tr( "Poles Set" ) );
		QObject::connect( button, SIGNAL(clicked()),
			this, SLOT( PolesSetKidneySegm()) );
		layout->addWidget( button );

		QSlider *slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setValue( 25 );
		slider->setMinimum( 5 );
		slider->setMaximum( 50 );
		KidneySegmentationManager::Instance().SetComputationPrecision( 25 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				&(KidneySegmentationManager::Instance()), SLOT( SetComputationPrecision( int ) ) );
		layout->addWidget( slider );


		button = new QPushButton( tr( "Start Segmentation" ) );
		QObject::connect( button, SIGNAL(clicked()),
			this, SLOT( StartSegmentationKidneySegm()) );
		layout->addWidget( button );

		layout->addStretch( 1 );

		button = new QPushButton( tr( "Process Results" ) );
		/*QObject::connect( button, SIGNAL(clicked()),
			this, SLOT( ProcessResultsManual()) );*/
		layout->addWidget( button );

		button = new QPushButton( tr( "Manual Correction" ) );
		QObject::connect( button, SIGNAL(clicked()),
			this, SLOT( BeginManualCorrection()) );
		layout->addWidget( button );

		layout->addStretch( 5 );
		
		_kidneySegmSettings->setLayout(layout);
	}
	addWidget( _kidneySegmSettings );
	
	QObject::connect( &(ManualSegmentationManager::Instance()), SIGNAL( StateUpdated() ),
			this, SLOT( ManualSegmentationUpdate() ) );

}

void
SettingsBox
::CreateManualGUI()
{
	QVBoxLayout *verticalLayout;
	QStackedWidget *stackedWidget;
	QWidget *page;
	QWidget *verticalLayoutWidget_2;
	QVBoxLayout *verticalLayout_2;
	QWidget *page_2;
	QWidget *verticalLayoutWidget_3;
	QVBoxLayout *verticalLayout_3;
	QPushButton *button;

	verticalLayout = new QVBoxLayout();
	
	QToolBar *tbar = new QToolBar();
	_createSplineButton = tbar->addAction( "New\nSpline" );
	_createSplineButton->setCheckable( true );
	QObject::connect( _createSplineButton, SIGNAL( toggled( bool ) ), &(ManualSegmentationManager::Instance()), SLOT( SetCreatingState( bool ) ) );

	_editPointsButton = tbar->addAction( "Edit\nPoints" );
	_editPointsButton->setCheckable( true );
	QObject::connect( _editPointsButton, SIGNAL( toggled( bool ) ), &(ManualSegmentationManager::Instance()), SLOT( SetEditPointsState( bool ) ) );
	
	/*action = tbar->addAction( "Edit\nSegs" );
	action->setCheckable( true );*/

	verticalLayout->addWidget( tbar );
	
	/*stackedWidget = new QStackedWidget();
	page = new QWidget();
	verticalLayoutWidget_2 = new QWidget(page);
	verticalLayout_2 = new QVBoxLayout(verticalLayoutWidget_2);
	stackedWidget->addWidget(page);
	page_2 = new QWidget();
	verticalLayoutWidget_3 = new QWidget(page_2);
	verticalLayout_3 = new QVBoxLayout(verticalLayoutWidget_3);
	stackedWidget->addWidget(page_2);

	verticalLayout->addWidget(stackedWidget);*/

	_deleteCurveButton = new QPushButton( tr( "Delete Curve" ) );
	QObject::connect( _deleteCurveButton, SIGNAL(clicked()),
		 &(ManualSegmentationManager::Instance()), SLOT( DeleteSelectedCurve()) );
	verticalLayout->addWidget( _deleteCurveButton );

	button = new QPushButton( tr( "Process Results" ) );
	QObject::connect( button, SIGNAL(clicked()),
		this, SLOT( ProcessResultsManual()) );
	verticalLayout->addWidget( button );

	verticalLayout->addStretch( 5 );

	_manualSegmSettings->setLayout(verticalLayout);
}

void
SettingsBox
::ManualSegmentationUpdate()
{
	switch( ManualSegmentationManager::Instance().GetInternalState() ) {
	case ManualSegmentationManager::SELECT:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( true );
		_editPointsButton->setEnabled( false );
		break;
	case ManualSegmentationManager::SELECTED:
		_deleteCurveButton->setEnabled( true );
		_createSplineButton->setEnabled( true );
		_editPointsButton->setEnabled( true );
		break;
	case ManualSegmentationManager::CREATING:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( true );
		_editPointsButton->setEnabled( false );
		break;
	case ManualSegmentationManager::SELECT_POINT:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( false );
		_editPointsButton->setEnabled( true );
		break;
	case ManualSegmentationManager::SELECTED_POINT:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( false );
		_editPointsButton->setEnabled( true );
		break;
	default:
		;
	}
}

void
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
}
