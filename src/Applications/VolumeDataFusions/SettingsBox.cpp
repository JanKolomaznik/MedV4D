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

	// create and add the widgets

	setMinimumWidth( MINIMUM_WIDTH );

	QVBoxLayout *layout;
	QGridLayout *grid;
	QButtonGroup *radioButtons;
	QRadioButton *rbutton;

	layout = new QVBoxLayout;

	clearButton = new QPushButton( tr( "Clear Selected Dataset" ) );
	layout->addWidget( clearButton );
	QObject::connect( clearButton, SIGNAL(clicked()),
		static_cast< mainWindow* >( _parent ), SLOT(ClearDataset()) );

	grid = new QGridLayout;

	grid->setRowMinimumHeight( 0, ROW_SPACING );

	//-------------------------------------------------
	
	grid->setRowMinimumHeight( 2, ROW_SPACING );
	grid->addWidget( new QLabel( tr( "Registration" ) ), 1, 1 );
	radioButtons = new QButtonGroup( this );
	rbutton = new QRadioButton( "Automatic" );
	grid->addWidget( rbutton, 1, 2 );
	radioButtons->addButton( rbutton, 0 );
	
	rbutton = new QRadioButton( "Manual" );
	rbutton->setChecked( true );
	grid->addWidget( rbutton, 1, 3 );
	radioButtons->addButton( rbutton, 1 );

	QObject::connect( radioButtons, SIGNAL( buttonClicked( int ) ), this, SLOT( RegistrationType( int ) ) );
	
	grid->addWidget( new QLabel( tr( "Rotation (degrees)" ) ), 2, 2 );
	grid->addWidget( new QLabel( tr( "Translation (pixels)" ) ), 2, 3 );

	grid->addWidget( new QLabel( tr( "X" ) ), 3, 1 );
	xRot = new QSpinBox();
	xRot->setAlignment( Qt::AlignRight );
	xRot->setMaximum( 360 );
	xRot->setMinimum( 0 );
	grid->addWidget(xRot, 3, 2 );

	xTrans = new QSpinBox();
	xTrans->setAlignment( Qt::AlignRight );
	xTrans->setMaximum( 250 );
	xTrans->setMinimum( -250 );
	xTrans->setValue( 0 );
	grid->addWidget(xTrans, 3, 3 );

	grid->addWidget( new QLabel( tr( "Y" ) ), 4, 1 );
	yRot = new QSpinBox();
	yRot->setAlignment( Qt::AlignRight );
	yRot->setMaximum( 360 );
	yRot->setMinimum( 0 );
	grid->addWidget(yRot, 4, 2 );

	yTrans = new QSpinBox();
	yTrans->setAlignment( Qt::AlignRight );
	yTrans->setMaximum( 250 );
	yTrans->setMinimum( -250 );
	yTrans->setValue( 0 );
	grid->addWidget(yTrans, 4, 3 );

	grid->addWidget( new QLabel( tr( "Z" ) ), 5, 1 );
	zRot = new QSpinBox();
	zRot->setAlignment( Qt::AlignRight );
	zRot->setMaximum( 360 );
	zRot->setMinimum( 0 );
	grid->addWidget(zRot, 5, 2 );

	zTrans = new QSpinBox();
	zTrans->setAlignment( Qt::AlignRight );
	zTrans->setMaximum( 250 );
	zTrans->setMinimum( -250 );
	zTrans->setValue( 0 );
	grid->addWidget(zTrans, 5, 3 );

	grid->addWidget( new QLabel( tr( "Registration Sampling" ) ), 6, 1, 1, 2 );
	accuracy = new QSpinBox();
	accuracy->setAlignment( Qt::AlignRight );
	accuracy->setMaximum( 2048 );
	accuracy->setMinimum( 2 );
	accuracy->setValue( TRANSFORM_SAMPLING );
	accuracy->setEnabled( false );
	grid->addWidget(accuracy, 6, 3 );
	
	grid->addWidget( new QLabel( tr( "Level of parallelization (number of processors)" ) ), 7, 1, 1, 2 );
	threadNum = new QSpinBox();
	threadNum->setAlignment( Qt::AlignRight );
	threadNum->setMaximum( 256 );
	threadNum->setMinimum( 1 );
	threadNum->setValue( 1 );
	grid->addWidget(threadNum, 7, 3 );
	

	//-------------------------------------------------

	interpolatorType = new QComboBox();
        interpolatorType->addItem( "Nearest Neighbor interpolator" );
        interpolatorType->addItem( "Linear interpolator" );

        interpolatorType->setCurrentIndex( 0 );

	//-------------------------------------------------

	fusionType = new QComboBox();

        fusionType->addItem( "RGB fusion" );
        fusionType->addItem( "Multiple colored fusion" );
        fusionType->addItem( "Maximum intensity fusion" );
        fusionType->addItem( "Minimum intensity fusion" );
        fusionType->addItem( "Average intensity fusion" );
        fusionType->addItem( "Median intensity fusion" );
        fusionType->addItem( "RGB-Max-Avr-Min fusion" );
        fusionType->addItem( "RGB-Max-Med-Min fusion" );
        fusionType->addItem( "Multiple colored gradient fusion" );
        fusionType->addItem( "RGB-Max-Avr-Min gradient fusion" );
        fusionType->addItem( "RGB-Max-Med-Min gradient fusion" );
        fusionType->addItem( "RGB-Max-Max-Min gradient fusion" );
        fusionType->addItem( "RGB-Max-Min-Min gradient fusion" );

        fusionType->setCurrentIndex( 0 );

	//-------------------------------------------------

	layout->addLayout( grid );
	layout->addWidget( interpolatorType );
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

	execButton = new QPushButton( tr( "Stop Single Registration" ) );
	QObject::connect( execButton, SIGNAL(clicked()),
			this, SLOT(StopSingleFilter()) );
	layout->addWidget(execButton);

	//-------------------------------------------------

	execButton = new QPushButton( tr( "Stop All Registrations" ) );
	QObject::connect( execButton, SIGNAL(clicked()),
			this, SLOT(StopAllFilters()) );
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
	return _viewers->getSelectedViewerSourceIdx() / 2;
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
	accuracy->setEnabled( !val );
}

void
SettingsBox
::ExecuteFilter( unsigned filterNum )
{

	// set the interpolator
	if ( interpolatorType->currentIndex() == 0 )	_registerFilters[ filterNum ]->SetInterpolator( &nearestNeighborInterpolator[ filterNum ] );
	else						_registerFilters[ filterNum ]->SetInterpolator( &linearInterpolator[ filterNum ] );

	// if manual registration is needed, set the parameters of the transformation
	if ( xRot->isEnabled() )
	{
		_registerFilters[ filterNum ]->SetAutomaticMode( false );
		float PI = std::atan(1.0f) * 4.0f;
		_registerFilters[ filterNum ]->SetRotation( InImageRegistration::CoordType( 2 * PI * xRot->value() / 360, 2 * PI * yRot->value() / 360, 2 * PI * zRot->value() / 360 ) );
		_registerFilters[ filterNum ]->SetTranslation( InImageRegistration::CoordType( xTrans->value(), yTrans->value(), zTrans->value() ) );
	}

	// if reference image is to be registered automatically, simply copy the image
	else if ( filterNum == 0 )
	{
		_registerFilters[ filterNum ]->SetAutomaticMode( false );
		_registerFilters[ filterNum ]->SetRotation( InImageRegistration::CoordType( 0, 0, 0 ) );
		_registerFilters[ filterNum ]->SetTranslation( InImageRegistration::CoordType( 0, 0, 0 ) );
	}

	// if automatic registration is requested, set automatic mode and sampling rate
	else
	{
		_registerFilters[ filterNum ]->SetAutomaticMode( true );
		_registerFilters[ filterNum ]->SetTransformSampling( accuracy->value() );
	}

	// set thread number and execute fusion
	_registerFilters[ filterNum ]->SetThreadNumber( threadNum->value() );
	_registerFilters[ filterNum ]->ExecuteOnWhole();
}

void
SettingsBox
::ExecSingleFilter()
{
	ExecuteFilter( GetInputNumber() );
}

void
SettingsBox
::ExecAllFilters()
{
	for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) ExecuteFilter( i );
}

void
SettingsBox
::StopSingleFilter()
{

	// execute registration filter with the number of the source selected
	_registerFilters[ GetInputNumber() ]->StopExecution();
}

void
SettingsBox
::StopAllFilters()
{
	for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) _registerFilters[ i ]->StopExecution();
}

void
SettingsBox
::ExecFusion()
{
	M4D::Viewer::m4dGUISliceViewerWidget* sliceViewer = dynamic_cast< M4D::Viewer::m4dGUISliceViewerWidget* >( _viewers->getSelectedViewerWidget() );
	if ( ! sliceViewer ) return;

	// select the fusion that is required
	for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i )
	{
		switch ( fusionType->currentIndex() )
		{

			case 0:
			sliceViewer->setTexturePreparerToRGB();
			break;

			case 1:
			sliceViewer->setTexturePreparerToCustom(&multiChannelRGBTexturePreparer);
			break;

			case 2:
			sliceViewer->setTexturePreparerToCustom(&maximumIntensityTexturePreparer);
			break;
			
			case 3:
			sliceViewer->setTexturePreparerToCustom(&minimumIntensityTexturePreparer);
			break;
			
			case 4:
			sliceViewer->setTexturePreparerToCustom(&averageIntensityTexturePreparer);
			break;
			
			case 5:
			sliceViewer->setTexturePreparerToCustom(&medianIntensityTexturePreparer);
			break;
			
			case 6:
			sliceViewer->setTexturePreparerToCustom(&maxAvrMinRGBTexturePreparer);
			break;
			
			case 7:
			sliceViewer->setTexturePreparerToCustom(&maxMedMinRGBTexturePreparer);
			break;

			case 8:
			sliceViewer->setTexturePreparerToCustom(&multiChannelGradientRGBTexturePreparer);
			break;

			case 9:
			sliceViewer->setTexturePreparerToCustom(&maxAvrMinGradientRGBTexturePreparer);
			break;

			case 10:
			sliceViewer->setTexturePreparerToCustom(&maxMedMinGradientRGBTexturePreparer);
			break;

			case 11:
			sliceViewer->setTexturePreparerToCustom(&maxMaxMinGradientRGBTexturePreparer);
			break;

			case 12:
			sliceViewer->setTexturePreparerToCustom(&maxMinMinGradientRGBTexturePreparer);
			break;

		}

		static_cast< mainWindow* >( _parent )->OutConnectionToViewerPort( i, i );
	}

	_viewers->UpdateViewers();
}

void
SettingsBox
::EndOfExecution( unsigned filterNum )
{

	// print message about the completition of registration
        QMessageBox::information( _parent, tr( "Execution finished" ), tr( "Registration Filter #%1 finished its work" ).arg( filterNum + 1 ) );
}
