#include "SettingsBox.h"
#include "Imaging/AbstractFilter.h"

SettingsBox
::SettingsBox( M4D::GUI::m4dGUIMainViewerDesktopWidget * viewers, InImageRegistration ** registers, QWidget * parent )
	: _viewers( viewers ), _registerFilters( registers ), _parent( parent )
{
	CreateWidgets();
}

void
SettingsBox
::CreateWidgets()
{
	setMinimumWidth( MINIMUM_WIDTH );

	QVBoxLayout *layout;
	QGridLayout *grid;
	QButtonGroup *radioButtons;
	QRadioButton *rbutton;

	layout = new QVBoxLayout;

	clearButton = new QPushButton( tr( "Clear Selected Dataset" ) );
	layout->addWidget( clearButton );
	QObject::connect( clearButton, SIGNAL(clicked()),
		this, SLOT(ClearDataset()) );

	grid = new QGridLayout;

	grid->setRowMinimumHeight( 0, ROW_SPACING );

	//-------------------------------------------------
	grid->addWidget( new QLabel( tr( "Fusion Order Number" ) ), 1, 1 );
	fusionNumber = new QSpinBox();
	fusionNumber->setAlignment( Qt::AlignRight );
	fusionNumber->setMaximum( SLICEVIEWER_INPUT_NUMBER );
	fusionNumber->setMinimum( 1 );
	grid->addWidget(fusionNumber, 1, 3 );
	//-------------------------------------------------
	
	grid->setRowMinimumHeight( 2, ROW_SPACING );
	grid->addWidget( new QLabel( tr( "Registration" ) ), 2, 1 );
	radioButtons = new QButtonGroup( this );
	rbutton = new QRadioButton( "Automatic" );
	grid->addWidget( rbutton, 2, 2 );
	radioButtons->addButton( rbutton, 0 );
	
	rbutton = new QRadioButton( "Manual" );
	rbutton->setChecked( true );
	grid->addWidget( rbutton, 2, 3 );
	radioButtons->addButton( rbutton, 1 );

	QObject::connect( radioButtons, SIGNAL( buttonClicked( int ) ), this, SLOT( RegistrationType( int ) ) );
	
	grid->addWidget( new QLabel( tr( "Rotation" ) ), 3, 2 );
	grid->addWidget( new QLabel( tr( "Translation" ) ), 3, 3 );

	grid->addWidget( new QLabel( tr( "X" ) ), 4, 1 );
	xRot = new QSpinBox();
	xRot->setAlignment( Qt::AlignRight );
	xRot->setMaximum( 360 );
	xRot->setMinimum( 0 );
	grid->addWidget(xRot, 4, 2 );

	xTrans = new QSpinBox();
	xTrans->setAlignment( Qt::AlignRight );
	xTrans->setMaximum( 250 );
	xTrans->setMinimum( -250 );
	xTrans->setValue( 0 );
	grid->addWidget(xTrans, 4, 3 );

	grid->addWidget( new QLabel( tr( "Y" ) ), 5, 1 );
	yRot = new QSpinBox();
	yRot->setAlignment( Qt::AlignRight );
	yRot->setMaximum( 360 );
	yRot->setMinimum( 0 );
	grid->addWidget(yRot, 5, 2 );

	yTrans = new QSpinBox();
	yTrans->setAlignment( Qt::AlignRight );
	yTrans->setMaximum( 250 );
	yTrans->setMinimum( -250 );
	yTrans->setValue( 0 );
	grid->addWidget(yTrans, 5, 3 );

	grid->addWidget( new QLabel( tr( "Z" ) ), 6, 1 );
	zRot = new QSpinBox();
	zRot->setAlignment( Qt::AlignRight );
	zRot->setMaximum( 360 );
	zRot->setMinimum( 0 );
	grid->addWidget(zRot, 6, 2 );

	zTrans = new QSpinBox();
	zTrans->setAlignment( Qt::AlignRight );
	zTrans->setMaximum( 250 );
	zTrans->setMinimum( -250 );
	zTrans->setValue( 0 );
	grid->addWidget(zTrans, 6, 3 );
	

	//-------------------------------------------------
	fusionType = new QComboBox();

        fusionType->addItem( "Simple" );
        fusionType->addItem( "RGB fusion" );
        fusionType->addItem( "Multiple colored fusion" );
        fusionType->addItem( "Maximum intensity fusion" );
        fusionType->addItem( "Average intensity fusion" );
        fusionType->addItem( "Minimum intensity fusion" );
        fusionType->addItem( "RGB-Max-Avr-Min fusion" );

        fusionType->setCurrentIndex( 0 );

	//-------------------------------------------------

	layout->addLayout( grid );
	layout->addWidget( fusionType );

	//-------------------------------------------------

	execButton = new QPushButton( tr( "Execute Single Registration" ) );
	QObject::connect( execButton, SIGNAL(clicked()),
			this, SLOT(ExecSingleFilter()) );
	layout->addWidget(execButton);

	//-------------------------------------------------

	execButton = new QPushButton( tr( "Execute All Registrations" ) );
	QObject::connect( execButton, SIGNAL(clicked()),
			this, SLOT(ExecAllFilters()) );
	layout->addWidget(execButton);

	//-------------------------------------------------

	layout->addSpacing( EXECUTE_BUTTON_SPACING );

	//-------------------------------------------------
	execButton = new QPushButton( tr( "Show Fusion" ) );
	QObject::connect( execButton, SIGNAL(clicked()),
                      	this, SLOT(ExecFusion()) );
	layout->addWidget(execButton);
	//-------------------------------------------------

	layout->addStretch();

	setLayout(layout);	
}

uint32
SettingsBox
::GetInputNumber()
{
	return fusionNumber->value();
}

void
SettingsBox
::ClearDataset()
{
}

void
SettingsBox
::RegistrationType( int val )
{
	xRot->setEnabled( val );
	xTrans->setEnabled( val );
	yRot->setEnabled( val );
	yTrans->setEnabled( val );
	zRot->setEnabled( val );
	zTrans->setEnabled( val );
}

void
SettingsBox
::ExecuteFilter( unsigned filterNum )
{
	if ( xRot->isEnabled() )
	{
		_registerFilters[ filterNum ]->SetAutomaticMode( false );
		float PI = std::atan(1.0f) * 4.0f;
		_registerFilters[ filterNum ]->SetRotation( InImageRegistration::CoordType( 2 * PI * xRot->value() / 360, 2 * PI * yRot->value() / 360, 2 * PI * zRot->value() / 360 ) );
		_registerFilters[ filterNum ]->SetTranslation( InImageRegistration::CoordType( xTrans->value(), yTrans->value(), zTrans->value() ) );
	}
	else _registerFilters[ filterNum ]->SetAutomaticMode( true );
	_registerFilters[ filterNum ]->ExecuteOnWhole();
}

void
SettingsBox
::ExecSingleFilter()
{
	ExecuteFilter( fusionNumber->value() - 1 );
}

void
SettingsBox
::ExecAllFilters()
{
	for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) ExecuteFilter( i );
}

void
SettingsBox
::ExecFusion()
{
}

void
SettingsBox
::EndOfExecution( unsigned filterNum )
{
        QMessageBox::information( _parent, tr( "Execution finished" ), tr( "Registration Filter #%1 finished its work" ).arg( filterNum ) );
}
