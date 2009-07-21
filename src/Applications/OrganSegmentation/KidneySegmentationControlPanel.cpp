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
		QLabel *label;
		QFrame *frame;

		_modelsCombo = new QComboBox();
		const ModelInfoVector & infos = _manager->GetModelInfos();
		for( unsigned i = 0; i < infos.size(); ++i ) {
			_modelsCombo->addItem( QString::fromStdString( infos[i].modelName ) );
		}
		QObject::connect( _modelsCombo, SIGNAL( currentIndexChanged( int ) ), 
				_manager, SLOT( SetModelID( int ) ) );
		_modelsCombo->setCurrentIndex( 0 );
		tmpLayout->addWidget( _modelsCombo );

		QSlider *slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 4 );
		slider->setMaximum( 35 );
		_manager->SetComputationPrecision( 15 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetComputationPrecision( int ) ) );
		slider->setValue( 15 );
		tmpLayout->addWidget( slider );

		//**************************************************
		//***** MODEL BALANCES *****************************
		frame = new QFrame;
		frame->setFrameShape( QFrame::HLine );
		tmpLayout->addWidget( frame );

		label = new QLabel( "Intensity distribution balance:" );
		tmpLayout->addWidget( label );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );
		_manager->SetDistBalance( 250 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetDistBalance( int ) ) );
		slider->setValue( 250 );
		tmpLayout->addWidget( slider );

		label = new QLabel( "Shape model balance:" );
		tmpLayout->addWidget( label );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );
		_manager->SetShapeBalance( 250 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetShapeBalance( int ) ) );
		slider->setValue( 250 );
		tmpLayout->addWidget( slider );

		label = new QLabel( "General model balance:" );
		tmpLayout->addWidget( label );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );
		_manager->SetGeneralBalance( 750 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetGeneralBalance( int ) ) );
		slider->setValue( 750 );
		tmpLayout->addWidget( slider );

		frame = new QFrame;
		frame->setFrameShape( QFrame::HLine );
		tmpLayout->addWidget( frame );
		//**************************************************
		//**************************************************
		label = new QLabel( "Edge importance:" );
		tmpLayout->addWidget( label );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );
		_manager->SetEdgeRegionBalance( 800 );
		QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetEdgeRegionBalance( int ) ) );
		slider->setValue( 800 );
		tmpLayout->addWidget( slider );

		/*label = new QLabel( "Internal energy balance:" );
		tmpLayout->addWidget( label );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );*/
		_manager->SetInternalEnergyBalance( 0 );
		/*QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetInternalEnergyBalance( int ) ) );
		slider->setValue( 0 );
		tmpLayout->addWidget( slider );

		label = new QLabel( "Internal energy gamma:" );
		tmpLayout->addWidget( label );

		slider = new QSlider();
		slider->setOrientation( Qt::Horizontal );
		slider->setMinimum( 0 );
		slider->setMaximum( 1000 );*/
		_manager->SetInternalEnergyGamma( 0 );
		/*QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
				_manager, SLOT( SetInternalEnergyGamma( int ) ) );
		slider->setValue( 0 );
		tmpLayout->addWidget( slider );

		QCheckBox *box = new QCheckBox( "Separate slice init" );
		box->setChecked( true );*/
		KidneySegmentationManager::Instance().SetSeparateSliceInit( true );
		/*QObject::connect( box, SIGNAL( toggled( bool ) ), 
				_manager, SLOT( SetSeparateSliceInit( bool ) ) );
		tmpLayout->addWidget( box );*/


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
