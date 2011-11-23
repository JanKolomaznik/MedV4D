#include "SettingsBox.h"
#include "mainWindow.h"

SettingsBox
::SettingsBox( M4D::Imaging::APipeFilter * filter, QWidget * parent )
	: _filter( filter ), _parent( parent )
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
	QPushButton *pushButton;

	layout = new QVBoxLayout;

	pushButton = new QPushButton( tr( "Lungs" ) );
	layout->addWidget( pushButton );
	QObject::connect( pushButton, SIGNAL(clicked()),
		this, SLOT(SetToLungs()) );

	pushButton = new QPushButton( tr( "Bones" ) );
	layout->addWidget( pushButton );
	QObject::connect( pushButton, SIGNAL(clicked()),
		this, SLOT(SetToBones()) );

	grid = new QGridLayout;

	grid->setRowMinimumHeight( 0, ROW_SPACING );

	//-------------------------------------------------
	grid->addWidget( new QLabel( tr( "Top" ) ), 1, 1 );
	top = new QSpinBox();
	top->setAlignment( Qt::AlignRight );
	top->setMaximum( 4095 );
	QObject::connect( top, SIGNAL(valueChanged(int)),
                      	this, SLOT(TopValueChanged(int)) );
	grid->addWidget(top, 1, 3 );
	//-------------------------------------------------
	
	grid->setRowMinimumHeight( 2, ROW_SPACING );

	//-------------------------------------------------
	grid->addWidget( new QLabel( tr( "Bottom" ) ), 3, 1 );
	bottom = new QSpinBox();
	bottom->setAlignment( Qt::AlignRight );
	bottom->setMaximum( 4095 );
	QObject::connect( bottom, SIGNAL(valueChanged(int)),
                      	this, SLOT(BottomValueChanged(int)) );
	grid->addWidget(bottom, 3, 3 );
	//-------------------------------------------------

	grid->setRowMinimumHeight( 4, ROW_SPACING );

	//-------------------------------------------------
	/*grid->addWidget( new QLabel( tr( "In value" ) ), 5, 1 );
	outValue = new QSpinBox();
	outValue->setAlignment( Qt::AlignRight );
	outValue->setMaximum( 4095 );
	QObject::connect( outValue, SIGNAL(valueChanged(int)),
                      	this, SLOT(OutValueChanged(int)) );
	grid->addWidget(outValue, 5, 3 );*/
	//-------------------------------------------------

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
::TopValueChanged( int val )
{
	static_cast<Thresholding*>(_filter)->SetTop( val );
}

void
SettingsBox
::BottomValueChanged( int val )
{
	static_cast<Thresholding*>(_filter)->SetBottom( val );
}

void
SettingsBox
::SetToLungs()
{
	top->setValue( 800 );
	bottom->setValue( 0 );
}

void
SettingsBox
::SetToBones()
{
	top->setValue( 4000 );
	bottom->setValue( 1150 );
}

void
SettingsBox
::ExecuteFilter()
{
	_filter->ExecuteOnWhole();
}

void
SettingsBox
::EndOfExecution()
{
	QMessageBox::information( _parent, tr( "Execution finished" ), tr( "Filter finished its work" ) );
}
