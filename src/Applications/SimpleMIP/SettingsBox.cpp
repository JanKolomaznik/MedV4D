#include "SettingsBox.h"
#include "mainWindow.h"

using namespace M4D::Imaging;

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
	QButtonGroup *radioButtons;
	QRadioButton *button;
	QComboBox *projectionCombo;

	layout = new QVBoxLayout;
	//-------------------------------------------------
	projectionCombo = new QComboBox();

	projectionCombo->addItem( "Maximum intensity" );
	projectionCombo->addItem( "Summed intensity" );
	projectionCombo->addItem( "Average intensity" );
	
	projectionCombo->setCurrentIndex( 0 );

	QObject::connect( projectionCombo, SIGNAL( currentIndexChanged( int ) ), this, SLOT( ChangeProjectionType( int ) ) );

	layout->addWidget( projectionCombo );

	radioButtons = new QButtonGroup( this );
	
	//layout->addWidget( radioButtons );
	//-------------------------------------------------
	
	button = new QRadioButton( "XY" );
	button->setChecked( true );
	static_cast<SimpleProjectionFilter*>(_filter)->SetPlane( XY_PLANE );
	static_cast<SimpleProjectionFilter*>(_filter)->SetProjectionType( PT_MAX );
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
	static_cast<SimpleProjectionFilter*>(_filter)->SetPlane( (CartesianPlanes)val );
}

void
SettingsBox
::ChangeProjectionType( int val )
{
	switch( val ) {
	case 0:
		static_cast<SimpleProjectionFilter*>(_filter)->SetProjectionType( M4D::Imaging::PT_MAX );
		break;
	case 1:
		static_cast<SimpleProjectionFilter*>(_filter)->SetProjectionType( M4D::Imaging::PT_SUM );
		break;
	case 2:
		static_cast<SimpleProjectionFilter*>(_filter)->SetProjectionType( M4D::Imaging::PT_AVERAGE );
		break;
	default:
		ASSERT( false );
	}
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
