#include "SettingsBox.h"

#include "mainWindow.h"


SettingsBox::SettingsBox ( M4D::Imaging::APipeFilter *filter, QWidget *parent )
	: filter( filter ), parent( parent )
{
	CreateWidgets();	
	SetEnabledExecButton( false );
}


void SettingsBox::CreateWidgets ()
{
	setMinimumWidth( MINIMUM_WIDTH );

	QVBoxLayout *layout = new QVBoxLayout;;

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
	
  QComboBox *interpolationCombo = new QComboBox();

	interpolationCombo->addItem( tr( "Nearest Neighbor Interpolator" ) );
	interpolationCombo->addItem( tr( "Linear Interpolator" ) );
	
	interpolationCombo->setCurrentIndex( 0 );

	connect( interpolationCombo, SIGNAL(currentIndexChanged( int )), this, SLOT(InterpolationTypeChanged( int )) );

	layout->addWidget( interpolationCombo );

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  layout->addSpacing( EXECUTE_BUTTON_SPACING );

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
  
  layout->addStretch();

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
	
  execButton = new QPushButton( tr( "Execute" ) );

	connect( execButton, SIGNAL(clicked()), this, SLOT(ExecuteFilter()) );

	layout->addWidget( execButton );
	
  // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

	setLayout( layout );	
}


void SettingsBox::InterpolationTypeChanged ( int val )
{
	switch( val ) 
  {
	  case 0:
		  static_cast< Registration * >(filter)->SetInterpolationType( M4D::Imaging::IT_NEAREST );
		  break;

	  case 1:
		  static_cast< Registration * >(filter)->SetInterpolationType( M4D::Imaging::IT_LINEAR );
		  break;

	  default:
		  ASSERT(false);
	}
}


void SettingsBox::ExecuteFilter ()
{
	filter->ExecuteOnWhole();
}


void SettingsBox::EndOfExecution ()
{
	QMessageBox::information( parent, tr( "Execution finished" ), tr( "Filter finished its work" ) );
}
