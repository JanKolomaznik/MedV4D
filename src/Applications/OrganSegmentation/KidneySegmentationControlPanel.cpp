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
	button = new QPushButton( tr( "Poles Set" ) );
	QObject::connect( button, SIGNAL(clicked()),
		_manager, SLOT( PolesSet()) );
	layout->addWidget( button );

	QSlider *slider = new QSlider();
	slider->setOrientation( Qt::Horizontal );
	slider->setMinimum( 4 );
	slider->setMaximum( 35 );
	slider->setValue( 15 );
	KidneySegmentationManager::Instance().SetComputationPrecision( 15 );
	QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
			_manager, SLOT( SetComputationPrecision( int ) ) );
	layout->addWidget( slider );

	slider = new QSlider();
	slider->setOrientation( Qt::Horizontal );
	slider->setMinimum( 0 );
	slider->setMaximum( 1000 );
	slider->setValue( 750 );
	KidneySegmentationManager::Instance().SetShapeIntensityBalance( 750 );
	QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
			_manager, SLOT( SetShapeIntensityBalance( int ) ) );
	layout->addWidget( slider );

	slider = new QSlider();
	slider->setOrientation( Qt::Horizontal );
	slider->setMinimum( 0 );
	slider->setMaximum( 1000 );
	slider->setValue( 800 );
	KidneySegmentationManager::Instance().SetEdgeRegionBalance( 800 );
	QObject::connect( slider, SIGNAL( valueChanged( int ) ), 
			_manager, SLOT( SetEdgeRegionBalance( int ) ) );
	layout->addWidget( slider );

	QCheckBox *box = new QCheckBox( "Separate slice init" );
	box->setChecked( true );
	KidneySegmentationManager::Instance().SetSeparateSliceInit( true );
	QObject::connect( box, SIGNAL( toggled( bool ) ), 
			_manager, SLOT( SetSeparateSliceInit( bool ) ) );
	layout->addWidget( box );


	button = new QPushButton( tr( "Start Segmentation" ) );
	QObject::connect( button, SIGNAL(clicked()),
		_manager, SLOT( StartSegmentation()) );
	layout->addWidget( button );

	layout->addStretch( 3 );

	button = new QPushButton( tr( "Manual Correction" ) );
	QObject::connect( button, SIGNAL(clicked()),
		_manager, SLOT( BeginManualCorrection()) );
	layout->addWidget( button );

	layout->addStretch( 1 );

	button = new QPushButton( tr( "Process Results" ) );
	QObject::connect( button, SIGNAL(clicked()), _manager, SLOT( WantProcessResults() ) );
	layout->addWidget( button );


	layout->addStretch( 5 );
	
	setLayout(layout);	
}

void
KidneySegmentationControlPanel
::PanelUpdate()
{
	
}
