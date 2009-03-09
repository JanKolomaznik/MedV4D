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

		layout->addStretch( 5 );
		
		_kidneySegmSettings->setLayout(layout);
	}
	addWidget( _kidneySegmSettings );

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

	verticalLayout = new QVBoxLayout();
	
	QToolBar *tbar = new QToolBar();
	tbar->addAction( "New\nSpline" );
	tbar->addAction( "Edit\nPoints" );
	tbar->addAction( "Edit\nSegs" );
	verticalLayout->addWidget( tbar );
	
	stackedWidget = new QStackedWidget();
	page = new QWidget();
	verticalLayoutWidget_2 = new QWidget(page);
	verticalLayout_2 = new QVBoxLayout(verticalLayoutWidget_2);
	stackedWidget->addWidget(page);
	page_2 = new QWidget();
	verticalLayoutWidget_3 = new QWidget(page_2);
	verticalLayout_3 = new QVBoxLayout(verticalLayoutWidget_3);
	stackedWidget->addWidget(page_2);

	verticalLayout->addWidget(stackedWidget);

	_manualSegmSettings->setLayout(verticalLayout);
}

void
SettingsBox
::SetToManualSegmentation()
{
	ManualSegmentationManager::Instance().Initialize();
	
	emit SetSegmentationSignal( stMANUAL );
	setCurrentWidget( _manualSegmSettings );
}

void
SettingsBox
::SetToKidneySegmentation()
{
	KidneySegmentationManager::Instance().Initialize();
	
	emit SetSegmentationSignal( stKIDNEYS );
	setCurrentWidget( _kidneySegmSettings );
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

