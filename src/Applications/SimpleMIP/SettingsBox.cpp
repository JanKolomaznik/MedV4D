#include "SettingsBox.h"
#include "mainWindow.h"

using namespace M4D::Imaging;

SettingsBox
::SettingsBox( M4D::Imaging::AbstractPipeFilter * filter )
	: _filter( filter )
{
	CreateWidgets();	
}

void
SettingsBox
::CreateWidgets()
{
	setMinimumWidth( MINIMUM_WIDTH );

	QVBoxLayout *layout;
	QButtonGroup *radioButtons;
	QRadioButton *button;

	layout = new QVBoxLayout;
	radioButtons = new QButtonGroup( this );
	
	//layout->addWidget( radioButtons );
	//-------------------------------------------------
	
	button = new QRadioButton( "XY" );
	button->setDown( true );
	static_cast<SimpleMIP*>(_filter)->SetPlane( XY_PLANE );
	layout->addWidget( button );
	radioButtons->addButton( button, XY_PLANE );
	
	button = new QRadioButton( "XZ" );
	layout->addWidget( button );
	radioButtons->addButton( button, XZ_PLANE );

	button = new QRadioButton( "YZ" );
	layout->addWidget( button );
	radioButtons->addButton( button, YZ_PLANE );

	QObject::connect( radioButtons, SIGNAL( buttonClicked( int ) ), this, SLOT( ChangeProjPlane( int ) ) );
	//-------------------------------------------------

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
::ChangeProjPlane( int val )
{
	static_cast<SimpleMIP*>(_filter)->SetPlane( (M4D::Imaging::CartesianPlanes)val );
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
	QMessageBox::information( this, tr( "Execution finished" ), tr( "Filter finished its work" ) );
}
