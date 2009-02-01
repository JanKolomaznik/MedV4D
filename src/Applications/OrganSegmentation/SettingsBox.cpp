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
		QVBoxLayout *layout;
		QPushButton *button;
		layout = new QVBoxLayout;
		button = new QPushButton( tr( "Finished" ) );
		/*QObject::connect( button, SIGNAL(clicked()),
			this, SLOT( SetToManualSegmentation()) );*/
		layout->addWidget( button );

		layout->addStretch( 5 );

		_manualSegmSettings->setLayout(layout);
	}
	addWidget( _manualSegmSettings );

	_kidneySegmSettings = new QWidget;
	{
		QVBoxLayout *layout;
		QPushButton *button;
		layout = new QVBoxLayout;
		button = new QPushButton( tr( "Finished" ) );
		QObject::connect( button, SIGNAL(clicked()),
			this, SLOT( FinishedUserInputKidneySeg()) );
		layout->addWidget( button );

		layout->addStretch( 5 );
		
		_kidneySegmSettings->setLayout(layout);
	}
	addWidget( _kidneySegmSettings );
//	grid = new QGridLayout;
//
//	grid->setRowMinimumHeight( 0, ROW_SPACING );
//
//	//-------------------------------------------------
//	grid->addWidget( new QLabel( tr( "Top" ) ), 1, 1 );
//	top = new QSpinBox();
//	top->setAlignment( Qt::AlignRight );
//	top->setMaximum( 4095 );
//	/*QObject::connect( top, SIGNAL(valueChanged(int)),
//                      	this, SLOT(TopValueChanged(int)) );
//	grid->addWidget(top, 1, 3 );*/
//	//-------------------------------------------------
//	
//	grid->setRowMinimumHeight( 2, ROW_SPACING );
//
//	//-------------------------------------------------
//	grid->addWidget( new QLabel( tr( "Bottom" ) ), 3, 1 );
//	bottom = new QSpinBox();
//	bottom->setAlignment( Qt::AlignRight );
//	bottom->setMaximum( 4095 );
///*	QObject::connect( bottom, SIGNAL(valueChanged(int)),
//                      	this, SLOT(BottomValueChanged(int)) );*/
//	grid->addWidget(bottom, 3, 3 );
//	//-------------------------------------------------
//
//	grid->setRowMinimumHeight( 4, ROW_SPACING );
//
//	//-------------------------------------------------
//	/*grid->addWidget( new QLabel( tr( "In value" ) ), 5, 1 );
//	outValue = new QSpinBox();
//	outValue->setAlignment( Qt::AlignRight );
//	outValue->setMaximum( 4095 );
//	QObject::connect( outValue, SIGNAL(valueChanged(int)),
//                      	this, SLOT(OutValueChanged(int)) );
//	grid->addWidget(outValue, 5, 3 );*/
//	//-------------------------------------------------
//
//	layout->addLayout( grid );
//
//	layout->addSpacing( EXECUTE_BUTTON_SPACING );
//
//	//-------------------------------------------------
//	execButton = new QPushButton( tr( "Execute" ) );
///*	QObject::connect( execButton, SIGNAL(clicked()),
//                      	this, SLOT(ExecuteFilter()) );*/
//	layout->addWidget(execButton);
//	//-------------------------------------------------
//
//	layout->addStretch();

}

void
SettingsBox
::SetToManualSegmentation()
{
	ManualSegmentationManager::Initialize();
	
	emit SetSegmentationSignal( stMANUAL );
	setCurrentWidget( _manualSegmSettings );
}

void
SettingsBox
::SetToKidneySegmentation()
{
	KidneySegmentationManager::Initialize();
	
	emit SetSegmentationSignal( stKIDNEYS );
	setCurrentWidget( _kidneySegmSettings );
}

void
SettingsBox
::FinishedUserInputKidneySeg()
{
	KidneySegmentationManager::UserInputFinished();

}

