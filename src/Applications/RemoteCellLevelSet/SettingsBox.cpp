
#include "Common.h"
#include "SettingsBox.h"
#include "mainWindow.h"

using namespace M4D::Imaging;

SettingsBox
::SettingsBox( RemoteFilterType *filter, LevelSetFilterProperties *props, 
		QWidget * parent )
	: filter_( filter ), props_(props), _parent( parent )
{
	CreateWidgets();
	SetEnabledExecButton( false );
}

void
SettingsBox
::CreateWidgets()
{
	setMinimumWidth( MINIMUM_WIDTH );

	QVBoxLayout *layout;
	QGridLayout *grid;

	layout = new QVBoxLayout;
	
	{
		grid = new QGridLayout;
		grid->setRowMinimumHeight( 0, ROW_SPACING );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "lowerThreshold" ) ), 1, 1 );
		lowerThreshold = new QSpinBox();
		lowerThreshold->setAlignment( Qt::AlignRight );
		lowerThreshold->setMaximum( -4095 );
		QObject::connect( lowerThreshold, SIGNAL(valueChanged(int)),
		                      	this, SLOT(lowerThresholdValueChanged(int)) );
		grid->addWidget(lowerThreshold, 1, 3 );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "upperThreshold" ) ), 2, 1 );
		upperThreshold = new QSpinBox();
		upperThreshold->setAlignment( Qt::AlignRight );
		upperThreshold->setMaximum( 4095 );
		QObject::connect( upperThreshold, SIGNAL(valueChanged(int)),
		                      	this, SLOT(upperThresholdValueChanged(int)) );
		grid->addWidget(upperThreshold, 2, 3 );		
		//-------------------------------------------------
		grid->setRowMinimumHeight( 3, ROW_SPACING );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "maxIterations" ) ), 4, 1 );
		maxIterations = new QSpinBox();
		maxIterations->setAlignment( Qt::AlignRight );
		maxIterations->setMaximum( 1200 );
		maxIterations->setValue(1000);
		QObject::connect( maxIterations, SIGNAL(valueChanged(int)),
		                      	this, SLOT(maxIterationsValueChanged(int)) );
		grid->addWidget(maxIterations, 4, 3 );
		//-------------------------------------------------
		grid->setRowMinimumHeight( 5, ROW_SPACING );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "initialDistance" ) ), 6, 1 );
		initialDistance = new QDoubleSpinBox();
		initialDistance->setAlignment( Qt::AlignRight );
		initialDistance->setRange( 4.0, 40.0 );
		initialDistance->setSingleStep( 0.5);
		initialDistance->setDecimals(1);
		QObject::connect( initialDistance, SIGNAL(valueChanged(double)),
		                      	this, SLOT(initialDistanceValueChanged(double)) );
		grid->addWidget(initialDistance, 6, 3 );
		//-------------------------------------------------
		grid->setRowMinimumHeight( 7, ROW_SPACING );
		//-------------------------------------------------		
		grid->addWidget( new QLabel( tr( "seedX" ) ), 8, 1 );
		seedX = new QSpinBox();
		seedX->setAlignment( Qt::AlignRight );
		seedX->setMinimum(0);
		seedX->setMaximum( 512 );
		seedX->setSingleStep(1);
		seedX->setValue(256);
		QObject::connect( seedX, SIGNAL(valueChanged(int)),
		                      	this, SLOT(seedXValueChanged(int)) );
		grid->addWidget(seedX, 8, 3 );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "seedY" ) ), 9, 1 );
		seedY = new QSpinBox();
		seedY->setAlignment( Qt::AlignRight );
		seedY->setMinimum(0);
		seedY->setMaximum( 512 );
		seedY->setSingleStep(1);
		seedY->setValue(256);
		QObject::connect( seedY, SIGNAL(valueChanged(int)),
		                      	this, SLOT(seedYValueChanged(int)) );
		grid->addWidget(seedY, 9, 3 );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "seedZ" ) ), 10, 1 );
		seedZ = new QSpinBox();
		seedZ->setAlignment( Qt::AlignRight );
		seedZ->setMinimum(0);
		seedZ->setMaximum( 512 );
		seedZ->setSingleStep(1);
		seedZ->setValue(256);
		QObject::connect( seedZ, SIGNAL(valueChanged(int)),
		                      	this, SLOT(seedZValueChanged(int)) );
		grid->addWidget(seedZ, 10, 3 );
		//-------------------------------------------------
		grid->setRowMinimumHeight( 11, ROW_SPACING );
		//-------------------------------------------------		
		grid->addWidget( new QLabel( tr( "curvatureScaling" ) ), 12, 1 );
		curvatureScaling = new QDoubleSpinBox();
		curvatureScaling->setAlignment( Qt::AlignRight );
		curvatureScaling->setRange( 0.0, 40.0 );
		curvatureScaling->setSingleStep( 0.5);
		curvatureScaling->setDecimals(1);
		QObject::connect( curvatureScaling, SIGNAL(valueChanged(double)),
		                      	this, SLOT(curvatureScalingValueChanged(double)) );
		grid->addWidget(curvatureScaling, 12, 3 );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "propagationScaling" ) ), 13, 1 );
		propagationScaling = new QDoubleSpinBox();
		propagationScaling->setAlignment( Qt::AlignRight );
		propagationScaling->setRange( 4.0, 40.0 );
		propagationScaling->setSingleStep( 0.5);
		propagationScaling->setDecimals(1);
		QObject::connect( propagationScaling, SIGNAL(valueChanged(double)),
		                      	this, SLOT(propagationScalingValueChanged(double)) );
		grid->addWidget(propagationScaling, 13, 3 );
		//-------------------------------------------------
		grid->addWidget( new QLabel( tr( "advectionScaling" ) ), 14, 1 );
		advectionScaling = new QDoubleSpinBox();
		advectionScaling->setAlignment( Qt::AlignRight );
		advectionScaling->setRange( 4.0, 40.0 );
		advectionScaling->setSingleStep( 0.5);
		advectionScaling->setDecimals(1);
		QObject::connect( advectionScaling, SIGNAL(valueChanged(double)),
		                      	this, SLOT(advectionScalingValueChanged(double)) );
		grid->addWidget(advectionScaling, 14, 3 );
		//-------------------------------------------------
		grid->setRowMinimumHeight( 15, ROW_SPACING );
	}
	layout->addLayout( grid );
	layout->addSpacing( EXECUTE_BUTTON_SPACING );

	//-------------------------------------------------
	execButton = new QPushButton( tr( "Execute" ) );
	QObject::connect( execButton, SIGNAL(clicked()),
                      	this, SLOT(ExecuteFilter()) );
	layout->addWidget(execButton);
	//-------------------------------------------------

	layout->addStretch();

	setLayout(layout);	
}

void
SettingsBox
::ExecuteFilter()
{
	filter_->ExecuteOnWhole();
}

void
SettingsBox
::EndOfExecution()
{
	QMessageBox::information( _parent, tr( "Execution finished" ), tr( "Filter finished its work" ) );
}
