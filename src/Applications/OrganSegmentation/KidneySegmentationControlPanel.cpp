#include "KidneySegmentationControlPanel.h"


KidneySegmentationControlPanel
::KidneySegmentationControlPanel( KidneySegmentationManager *manager ): QWidget(), _manager( manager )
{
	CreateWidgets();
}

void
KidneySegmentationControlPanel
::CreateWidgets()
{
	QVBoxLayout *layout;
	QPushButton *button;
	layout = new QVBoxLayout;
	setLayout(layout);	

	_polesButtons = new QFrame();
	_polesButtons->setFrameStyle( QFrame::Box | QFrame::Plain );
	layout->addWidget( _polesButtons );
	{
		QVBoxLayout *tmpLayout = new QVBoxLayout;

		button = new QPushButton( tr( "Set new poles" ) );
		QObject::connect( button, SIGNAL(clicked()),
			_manager, SLOT( SetNewPoles()) );
		tmpLayout->addWidget( button );
		button = new QPushButton( tr( "Poles set" ) );
		QObject::connect( button, SIGNAL(clicked()),
			_manager, SLOT( PolesSet()) );
		tmpLayout->addWidget( button );

		_polesButtons->setLayout( tmpLayout );
	}

	
	_segmentationSettings = new QFrame();
	_segmentationSettings->setFrameStyle( QFrame::Box | QFrame::Plain );
	layout->addWidget( _segmentationSettings );
	{
		QVBoxLayout *tmpLayout = new QVBoxLayout;

		QSlider *slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 4 );
		slider->setMaximum( 35 );
		slider->setValue( 15 );
		KidneySegmentationManager::Instance().SetComputationPrecision( 15 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetComputationPrecision( int ) ) );
		tmpLayout->addWidget( slider );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );
		slider->setValue( 750 );
		KidneySegmentationManager::Instance().SetShapeIntensityBalance( 750 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetShapeIntensityBalance( int ) ) );
		tmpLayout->addWidget( slider );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );
		slider->setValue( 800 );
		KidneySegmentationManager::Instance().SetEdgeRegionBalance( 800 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetEdgeRegionBalance( int ) ) );
		tmpLayout->addWidget( slider );

		QCheckBox *box = new QCheckBox( "Separate slice init" );
		box->setChecked( true );
		KidneySegmentationManager::Instance().SetSeparateSliceInit( true );
		QObject::connect( box, SIGNAL( toggled( bool ) ), 
				_manager, SLOT( SetSeparateSliceInit( bool ) ) );
		tmpLayout->addWidget( box );


		button = new QPushButton( tr( "Start segmentation" ) );
		QObject::connect( button, SIGNAL(clicked()),
			_manager, SLOT( StartSegmentation()) );
		tmpLayout->addWidget( button );

		_segmentationSettings->setLayout( tmpLayout );
	}


	//layout->addStretch( 3 );
	_resultButtons = new QFrame();
	_resultButtons->setFrameStyle( QFrame::Box | QFrame::Plain );
	layout->addWidget( _resultButtons );
	{
		QVBoxLayout *tmpLayout = new QVBoxLayout;

		button = new QPushButton( tr( "Manual correction" ) );
		QObject::connect( button, SIGNAL(clicked()),
			_manager, SLOT( BeginManualCorrection()) );
		tmpLayout->addWidget( button );

		tmpLayout->addStretch( 1 );

		button = new QPushButton( tr( "Process results" ) );
		QObject::connect( button, SIGNAL(clicked()), _manager, SLOT( WantsProcessResults() ) );
		tmpLayout->addWidget( button );

		_resultButtons->setLayout( tmpLayout );
	}

	_infoBox = new QLabel();
	_infoBox->setWordWrap( true );
	layout->addWidget( _infoBox );



	QObject::connect( _manager, SIGNAL(StateUpdated()), this, SLOT( PanelUpdate()), Qt::QueuedConnection );

	layout->addStretch( 5 );
	
}

void
KidneySegmentationControlPanel
::PanelUpdate()
{
	switch( _manager->GetInternalState() ) {
	case KidneySegmentationManager::DEFINING_POLE:
		_polesButtons->setEnabled( true );
		_segmentationSettings->setEnabled( false );
		_resultButtons->setEnabled( false );
		_infoBox->setText( "Define poles positions..." );
		break;
	case KidneySegmentationManager::POLES_SET:
		_polesButtons->setEnabled( true );
		_segmentationSettings->setEnabled( false );
		_resultButtons->setEnabled( false );
		_infoBox->setText( "Poles set - confirm or set new positions." );
		break;
	case KidneySegmentationManager::SET_SEGMENTATION_PARAMS_PREPROCESSING:
		_infoBox->setText( "Set parameters\nPreprocessing inputs..." );
		_polesButtons->setEnabled( false );
		_segmentationSettings->setEnabled( true );
		_resultButtons->setEnabled( false );
		break;
	case KidneySegmentationManager::SET_SEGMENTATION_PARAMS:
		_infoBox->setText( "Set parameters" );
		_polesButtons->setEnabled( false );
		_segmentationSettings->setEnabled( true );
		_resultButtons->setEnabled( false );
		break;
	case KidneySegmentationManager::SEGMENTATION_EXECUTED_WAITING:
		_infoBox->setText( "Segmentation executed\nWaiting till preprocessing is finished" );
		_polesButtons->setEnabled( false );
		_segmentationSettings->setEnabled( false );
		_resultButtons->setEnabled( false );
		break;
	case KidneySegmentationManager::SEGMENTATION_EXECUTED_RUNNING:
		_infoBox->setText( "Segmentation executed - running..." );
		_polesButtons->setEnabled( false );
		_segmentationSettings->setEnabled( false );
		_resultButtons->setEnabled( false );
		break;
	case KidneySegmentationManager::SEGMENTATION_FINISHED:
		_infoBox->setText( "Segmentation finished" );
		_polesButtons->setEnabled( false );
		_segmentationSettings->setEnabled( true );
		_resultButtons->setEnabled( true );
		break;
	default:
		ASSERT( false );
	}
	
}
