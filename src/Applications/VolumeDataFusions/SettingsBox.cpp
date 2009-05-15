#include "SettingsBox.h"
#include "mainWindow.h"

SettingsBox
::SettingsBox( M4D::GUI::m4dGUIMainViewerDesktopWidget * viewers, QWidget * parent )
	: _viewers( viewers ), _parent( parent )
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
	QObject::connect( xRot, SIGNAL(valueChanged(int)),
                      	this, SLOT(xRotValueChanged(int)) );
	grid->addWidget(xRot, 4, 2 );

	xTrans = new QSpinBox();
	xTrans->setAlignment( Qt::AlignRight );
	xTrans->setMaximum( 250 );
	xTrans->setMinimum( -250 );
	xTrans->setValue( 0 );
	QObject::connect( xTrans, SIGNAL(valueChanged(int)),
                      	this, SLOT(xTransValueChanged(int)) );
	grid->addWidget(xTrans, 4, 3 );

	grid->addWidget( new QLabel( tr( "Y" ) ), 5, 1 );
	yRot = new QSpinBox();
	yRot->setAlignment( Qt::AlignRight );
	yRot->setMaximum( 360 );
	yRot->setMinimum( 0 );
	QObject::connect( yRot, SIGNAL(valueChanged(int)),
                      	this, SLOT(yRotValueChanged(int)) );
	grid->addWidget(yRot, 5, 2 );

	yTrans = new QSpinBox();
	yTrans->setAlignment( Qt::AlignRight );
	yTrans->setMaximum( 250 );
	yTrans->setMinimum( -250 );
	yTrans->setValue( 0 );
	QObject::connect( yTrans, SIGNAL(valueChanged(int)),
                      	this, SLOT(yTransValueChanged(int)) );
	grid->addWidget(yTrans, 5, 3 );

	grid->addWidget( new QLabel( tr( "Z" ) ), 6, 1 );
	zRot = new QSpinBox();
	zRot->setAlignment( Qt::AlignRight );
	zRot->setMaximum( 360 );
	zRot->setMinimum( 0 );
	QObject::connect( zRot, SIGNAL(valueChanged(int)),
                      	this, SLOT(zRotValueChanged(int)) );
	grid->addWidget(zRot, 6, 2 );

	zTrans = new QSpinBox();
	zTrans->setAlignment( Qt::AlignRight );
	zTrans->setMaximum( 250 );
	zTrans->setMinimum( -250 );
	zTrans->setValue( 0 );
	QObject::connect( zTrans, SIGNAL(valueChanged(int)),
                      	this, SLOT(zTransValueChanged(int)) );
	grid->addWidget(zTrans, 6, 3 );
	

	//-------------------------------------------------
	fusionType = new QComboBox();

        fusionType->addItem( "Maximum intensity" );
        fusionType->addItem( "Summed intensity" );
        fusionType->addItem( "Average intensity" );

        fusionType->setCurrentIndex( 0 );

	//-------------------------------------------------

	layout->addLayout( grid );
	layout->addWidget( fusionType );

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
::xRotValueChanged( int val )
{
}

void
SettingsBox
::xTransValueChanged( int val )
{
}

void
SettingsBox
::yRotValueChanged( int val )
{
}

void
SettingsBox
::yTransValueChanged( int val )
{
}

void
SettingsBox
::zRotValueChanged( int val )
{
}

void
SettingsBox
::zTransValueChanged( int val )
{
}

void
SettingsBox
::RegistrationType( int val )
{
}

void
SettingsBox
::ExecFusion()
{
}
