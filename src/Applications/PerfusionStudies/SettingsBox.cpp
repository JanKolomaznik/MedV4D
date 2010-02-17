#include "SettingsBox.h"

#include "mainWindow.h"


SettingsBox::SettingsBox ( M4D::Imaging::APipeFilter *registrationFilter,
                           M4D::Imaging::APipeFilter *segmentationFilter,
                           M4D::Imaging::APipeFilter *analysisFilter,
                           QWidget *parent )
	: registrationFilter( registrationFilter ), segmentationFilter( segmentationFilter ), analysisFilter( analysisFilter ),
    parent( parent )
{
	CreateWidgets();	
	
  execButton->setEnabled( false );
  cancelButton->setEnabled( false );

  onceFinished = false;
}


void SettingsBox::CreateWidgets ()
{
	setMinimumWidth( MINIMUM_WIDTH );

	QVBoxLayout *layout = new QVBoxLayout;

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QTabWidget *settingsTab = new QTabWidget;

  // -------------------------------------------

  QGridLayout *generalGrid = new QGridLayout;

  generalGrid->addWidget( new QLabel( tr( "Interpolation type" ) ), 0, 0 );
	
  QComboBox *interpolationCombo = new QComboBox();
	interpolationCombo->addItem( tr( "Nearest Neighbor" ) );
	interpolationCombo->addItem( tr( "Linear" ) );
	interpolationCombo->setCurrentIndex( 0 );

	connect( interpolationCombo, SIGNAL(currentIndexChanged( int )), this, SLOT(InterpolationTypeChanged( int )) );

  generalGrid->addWidget( interpolationCombo, 0, 1 );

  generalGrid->addWidget( new QLabel( tr( "Background" ) ), 1, 0 );
	
  QSpinBox *backgroundSpin = new QSpinBox();
	backgroundSpin->setAlignment( Qt::AlignRight );
	backgroundSpin->setMaximum( 4095 );
  backgroundSpin->setValue( BACKGROUND );
	
  connect( backgroundSpin, SIGNAL(valueChanged( int )), this, SLOT(BackgroundValueChanged( int )) );
	
  generalGrid->addWidget( backgroundSpin, 1, 1 );

  QWidget *generalPane = new QWidget;
  generalPane->setLayout( generalGrid );
  settingsTab->addTab( generalPane, tr( "General" ) );

  // -------------------------------------------

  QGridLayout *advancedGrid = new QGridLayout;

  advancedGrid->addWidget( new QLabel( tr( "Examined slice number" ) ), 0, 0 );
	
  QSpinBox *sliceNumberSpin = new QSpinBox();
	sliceNumberSpin->setAlignment( Qt::AlignRight );
  sliceNumberSpin->setMinimum( 1 );
	sliceNumberSpin->setMaximum( 1024 );
  sliceNumberSpin->setValue( EXEMINED_SLICE_NUM );
	
  connect( sliceNumberSpin, SIGNAL(valueChanged( int )), this, SLOT(SliceNumberValueChanged( int )) );
	
  advancedGrid->addWidget( sliceNumberSpin, 0, 2 );

  advancedGrid->addWidget( new QLabel( tr( "Bone density interval" ) ), 1, 0 );
	
  QSpinBox *boneDensityBottomSpin = new QSpinBox();
	boneDensityBottomSpin->setAlignment( Qt::AlignRight );
	boneDensityBottomSpin->setMaximum( 4095 );
  boneDensityBottomSpin->setValue( BONE_DENSITY_BOTTOM );
	
  connect( boneDensityBottomSpin, SIGNAL(valueChanged( int )), this, SLOT(BoneDensityBottomValueChanged( int )) );
	
  advancedGrid->addWidget( boneDensityBottomSpin, 1, 1 );

  QSpinBox *boneDensityTopSpin = new QSpinBox();
	boneDensityTopSpin->setAlignment( Qt::AlignRight );
	boneDensityTopSpin->setMaximum( 4095 );
  boneDensityTopSpin->setValue( BONE_DENSITY_TOP );
	
  connect( boneDensityTopSpin, SIGNAL(valueChanged( int )), this, SLOT(BoneDensityTopValueChanged( int )) );
	
  advancedGrid->addWidget( boneDensityTopSpin, 1, 2 );

  QWidget *advancedPane = new QWidget;
  advancedPane->setLayout( advancedGrid );
  settingsTab->addTab( advancedPane, tr( "Advanced" ) );

  // -------------------------------------------

  layout->addWidget( settingsTab );

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  layout->addSpacing( SPACING );

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QGroupBox *visualizationGroupBox = new QGroupBox( tr( "Visalization and analysis types" ) );

  QGridLayout *visualizationGrid = new QGridLayout;
  visualizationGrid->setColumnMinimumWidth( 0, TOOL_BUTTON_SIZE + 15 );

  // -------------------------------------------

  visualizationGrid->addWidget( new QLabel( tr( "Maximum intensity projection" ) ), 0, 0, 1, 5 );

  QToolButton *maximumButton = CreateToolButton( QIcon( ":/resources/maximum.png" ), 
                                                 SLOT(MaximumTypeSet()) );
  maximumButton->setChecked( true );

  visualizationGrid->addWidget( maximumButton, 1, 0, Qt::AlignLeft );

	// -------------------------------------------

  visualizationGrid->addWidget( new QLabel( tr( "Subtraction images" ) ), 2, 0, 1, 5 );
  
  QToolButton *subtractionButton = CreateToolButton( QIcon( ":/resources/subtraction.png" ), 
                                                     SLOT(SubtractionTypeSet()) );

  visualizationGrid->addWidget( subtractionButton, 3, 0, 2, 1, Qt::AlignLeft );

  defaultSubtractionCheckBox = CreateCheckBox( tr( "max CA - before CA arrival" ), true, SLOT(DefaultSubtractionChanged( bool )) );
  defaultSubtractionCheckBox->setEnabled( false );

  connect( defaultSubtractionCheckBox, SIGNAL(clicked()), subtractionButton, SLOT(click()) );
  
  visualizationGrid->addWidget( defaultSubtractionCheckBox, 3, 1, 1, 5 );

  visualizationGrid->addWidget( new QLabel( tr( "Indices" ) ), 4, 1 );
  
  highIndexSpin = new QSpinBox();
	highIndexSpin->setAlignment( Qt::AlignRight );
  highIndexSpin->setEnabled( false );
	
  connect( highIndexSpin, SIGNAL(valueChanged( int )), this, SLOT(HighIndexChanged( int )) );
  connect( defaultSubtractionCheckBox, SIGNAL(clicked( bool )), highIndexSpin, SLOT(setDisabled( bool )) );
  connect( highIndexSpin, SIGNAL(editingFinished()), subtractionButton, SLOT(click()) );
	
  visualizationGrid->addWidget( highIndexSpin, 4, 2 );

  visualizationGrid->addWidget( new QLabel( tr( "-" ) ), 4, 3 );

  lowIndexSpin = new QSpinBox();
	lowIndexSpin->setAlignment( Qt::AlignRight );
  lowIndexSpin->setEnabled( false );
	
  connect( lowIndexSpin, SIGNAL(valueChanged( int )), this, SLOT(LowIndexChanged( int )) );
  connect( defaultSubtractionCheckBox, SIGNAL(clicked( bool )), lowIndexSpin, SLOT(setDisabled( bool )) );
  connect( lowIndexSpin, SIGNAL(editingFinished()), subtractionButton, SLOT(click()) );
	
  visualizationGrid->addWidget( lowIndexSpin, 4, 4 );

	// -------------------------------------------

  visualizationGrid->addWidget( new QLabel( tr( "Parameter maps" ) ), 5, 0, 1, 5 );

  QToolButton *parameterButton = CreateToolButton( QIcon( ":/resources/parameter.png" ), 
                                                   SLOT(ParameterTypeSet()) );

  visualizationGrid->addWidget( parameterButton, 6, 0, Qt::AlignLeft );

  QGridLayout *parameterTypesGrid = new QGridLayout;

  // ----------------

  TTPRadio = CreateRadioButton( tr( "TTP" ), true, SLOT(TTPParameterSet()) );

  connect( TTPRadio, SIGNAL(clicked()), parameterButton, SLOT(click()) );

  parameterTypesGrid->addWidget( TTPRadio, 0, 0 );

  // ----------------
  
  ATRadio = CreateRadioButton( tr( "AT" ), false, SLOT(ATParameterSet()) );

  connect( ATRadio, SIGNAL(clicked()), parameterButton, SLOT(click()) );

  parameterTypesGrid->addWidget( ATRadio, 0, 1 );

  // ----------------
  
  CBVRadio = CreateRadioButton( tr( "CBV" ), false, SLOT(CBVParameterSet()) );

  connect( CBVRadio, SIGNAL(clicked()), parameterButton, SLOT(click()) );

  parameterTypesGrid->addWidget( CBVRadio, 1, 0 );

  // ----------------
  
  MTTRadio = CreateRadioButton( tr( "MTT" ), false, SLOT(MTTParameterSet()) );

  connect( MTTRadio, SIGNAL(clicked()), parameterButton, SLOT(click()) );

  parameterTypesGrid->addWidget( MTTRadio, 1, 1 );

  // ----------------

  CBFRadio = CreateRadioButton( tr( "CBF" ), false, SLOT(CBFParameterSet()) );

  connect( CBFRadio, SIGNAL(clicked()), parameterButton, SLOT(click()) );

  parameterTypesGrid->addWidget( CBFRadio, 1, 2 );

  // ---------------- 
  
  visualizationGrid->addLayout( parameterTypesGrid, 6, 1, 1, 5 );

  maxValuePercentageSlider = new QSlider( Qt::Horizontal );
  maxValuePercentageSlider->setValue( TTP_MAX_PERCENTAGE * 100 );

  connect( maxValuePercentageSlider, SIGNAL(valueChanged( int )), this, SLOT(MaxValuePercentageChanged( int )) );
  connect( maxValuePercentageSlider, SIGNAL(sliderReleased()), parameterButton, SLOT(click()) );

  visualizationGrid->addWidget( maxValuePercentageSlider, 7, 1, 1, 5 );

  // -------------------------------------------

  visualizationGroupBox->setLayout( visualizationGrid ); 
  
  layout->addWidget( visualizationGroupBox );

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
  
  layout->addSpacing( SPACING );

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
  
  layout->addStretch();

	// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
	
  QHBoxLayout *buttonLayout = new QHBoxLayout;
  
  execButton = new QPushButton( tr( "Execute" ) );

	connect( execButton, SIGNAL(clicked()), this, SLOT(ExecuteFilter()) );

	buttonLayout->addWidget( execButton );

  cancelButton = new QPushButton( tr( "Cancel" ) );

	connect( cancelButton, SIGNAL(clicked()), this, SLOT(CancelFilter()) );

	buttonLayout->addWidget( cancelButton );

  layout->addLayout( buttonLayout );
	
  // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

	setLayout( layout );	
}


QToolButton *SettingsBox::CreateToolButton ( const QIcon &icon, const char *member )
{
  QToolButton *toolButton = new QToolButton();

  toolButton->setCheckable( true );
  toolButton->setAutoExclusive( true );

  toolButton->setIconSize( QSize( TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE ) );
  toolButton->setIcon( icon );

  connect( toolButton, SIGNAL(clicked()), this, member );
  connect( toolButton, SIGNAL(clicked()), toolButton, SLOT(down()) );

  return toolButton;
}


QCheckBox *SettingsBox::CreateCheckBox ( const QString &text, bool checked, const char *member )
{
  QCheckBox *checkBox = new QCheckBox( text );

  checkBox->setChecked( checked );

  connect( checkBox, SIGNAL(clicked( bool )), this, member );
  
  return checkBox;
}


QRadioButton *SettingsBox::CreateRadioButton ( const QString &text, bool checked, const char *member )
{
  QRadioButton *radio = new QRadioButton( text );

  radio->setAutoExclusive( false );
  radio->setChecked( checked );

  connect( radio, SIGNAL(clicked()), this, member );
  
  return radio;
}


void SettingsBox::InterpolationTypeChanged ( int val )
{
	switch( val ) 
  {
	  case 0:
		  static_cast< Registration * >( registrationFilter )->SetInterpolationType( M4D::Imaging::IT_NEAREST );
		  break;

	  case 1:
		  static_cast< Registration * >( registrationFilter )->SetInterpolationType( M4D::Imaging::IT_LINEAR );
		  break;

	  default:
		  ASSERT(false);
	}
}


void SettingsBox::BackgroundValueChanged ( int val )
{
  static_cast< Segmentation * >( segmentationFilter )->SetBackground( val );
}


void SettingsBox::SliceNumberValueChanged ( int val )
{
  static_cast< Registration * >( registrationFilter )->SetExaminedSliceNum( val );
  static_cast< Segmentation * >( segmentationFilter )->SetExaminedSliceNum( val );
  static_cast< Analysis * >( analysisFilter )->SetExaminedSliceNum( val );
}


void SettingsBox::BoneDensityBottomValueChanged ( int val )
{
  static_cast< Registration * >( registrationFilter )->SetBoneDensityBottom( val );
  static_cast< Segmentation * >( segmentationFilter )->SetBoneDensityBottom( val );
}


void SettingsBox::BoneDensityTopValueChanged ( int val )
{
  static_cast< Registration * >( registrationFilter )->SetBoneDensityTop( val );
}


void SettingsBox::MaximumTypeSet ()
{
  emit SimpleSelected();

  static_cast< Analysis * >( analysisFilter )->SetVisualizationType( M4D::Imaging::VT_MAX );  

  if ( onceFinished ) 
  {
    static_cast< Analysis * >( analysisFilter )->Visualize();  

    emit VisualizationDone();
  }
}


void SettingsBox::SubtractionTypeSet ()
{
  emit SimpleSelected();

  static_cast< Analysis * >( analysisFilter )->SetVisualizationType( M4D::Imaging::VT_SUBTR );

  if ( onceFinished )
  {
    static_cast< Analysis * >( analysisFilter )->Visualize();  
    
    emit VisualizationDone();
  }
}


void SettingsBox::ParameterTypeSet ()
{
  static_cast< Analysis * >( analysisFilter )->SetVisualizationType( M4D::Imaging::VT_PARAM );  

  if ( onceFinished ) {
    static_cast< Analysis * >( analysisFilter )->Visualize();  
  }

  emit ParamaterMapsSelected();
}


void SettingsBox::DefaultSubtractionChanged ( bool checked )
{
  if ( checked ) 
  {
    static_cast< Analysis * >( analysisFilter )->SetSubtrIndexLow( -1 );
    static_cast< Analysis * >( analysisFilter )->SetSubtrIndexHigh( -1 );
  }
  else
  {
    static_cast< Analysis * >( analysisFilter )->SetSubtrIndexLow( lowIndexSpin->value() );
    static_cast< Analysis * >( analysisFilter )->SetSubtrIndexHigh( highIndexSpin->value() );   
  }
}


void SettingsBox::LowIndexChanged ( int val )
{
  static_cast< Analysis * >( analysisFilter )->SetSubtrIndexLow( val );
}


void SettingsBox::HighIndexChanged ( int val )
{
  static_cast< Analysis * >( analysisFilter )->SetSubtrIndexHigh( val );
}


void SettingsBox::TTPParameterSet ()
{
  // simulating autoexlusivity
  TTPRadio->setChecked( true );
  
  ATRadio->setChecked( false );
  CBVRadio->setChecked( false );
  MTTRadio->setChecked( false );
  CBFRadio->setChecked( false );

  maxValuePercentageSlider->setValue( TTP_MAX_PERCENTAGE * 100 );

  static_cast< Analysis * >( analysisFilter )->SetParameterType( M4D::Imaging::PT_TTP );
}


void SettingsBox::ATParameterSet ()
{
  ATRadio->setChecked( true );
  
  TTPRadio->setChecked( false );
  CBVRadio->setChecked( false );
  MTTRadio->setChecked( false );
  CBFRadio->setChecked( false );

  maxValuePercentageSlider->setValue( AT_MAX_PERCENTAGE * 100 );

  static_cast< Analysis * >( analysisFilter )->SetParameterType( M4D::Imaging::PT_AT );
}


void SettingsBox::CBVParameterSet ()
{
  CBVRadio->setChecked( true );
  
  TTPRadio->setChecked( false );
  ATRadio->setChecked( false );
  MTTRadio->setChecked( false );
  CBFRadio->setChecked( false );

  maxValuePercentageSlider->setValue( CBV_MAX_PERCENTAGE * 100 );

  static_cast< Analysis * >( analysisFilter )->SetParameterType( M4D::Imaging::PT_CBV );
}


void SettingsBox::MTTParameterSet ()
{
  MTTRadio->setChecked( true );
  
  TTPRadio->setChecked( false );
  ATRadio->setChecked( false );
  CBVRadio->setChecked( false );
  CBFRadio->setChecked( false );

  maxValuePercentageSlider->setValue( MTT_MAX_PERCENTAGE * 100 );

  static_cast< Analysis * >( analysisFilter )->SetParameterType( M4D::Imaging::PT_MTT );
}


void SettingsBox::CBFParameterSet ()
{
  CBFRadio->setChecked( true );
  
  TTPRadio->setChecked( false );
  ATRadio->setChecked( false );
  CBVRadio->setChecked( false );
  MTTRadio->setChecked( false );

  maxValuePercentageSlider->setValue( CBF_MAX_PERCENTAGE * 100 );

  static_cast< Analysis * >( analysisFilter )->SetParameterType( M4D::Imaging::PT_CBF );
}


void SettingsBox::MaxValuePercentageChanged ( int value )
{
  static_cast< Analysis * >( analysisFilter )->SetMaxValuePercentage( (float32)value / 100 );
}


void SettingsBox::ExecuteFilter ()
{
  execButton->setEnabled( false );
  cancelButton->setEnabled( true );

	registrationFilter->ExecuteOnWhole();
}


void SettingsBox::CancelFilter ()
{
  cancelButton->setEnabled( false );
  execButton->setEnabled( true );

  registrationFilter->StopExecution();
  segmentationFilter->StopExecution();
  analysisFilter->StopExecution();
}


void SettingsBox::EndOfExecution ()
{
  cancelButton->setEnabled( false );
  execButton->setEnabled( true );

  defaultSubtractionCheckBox->setEnabled( true );

  lowIndexSpin->setValue( 0 );
  highIndexSpin->setValue( static_cast< Analysis * >( analysisFilter )->GetMaxCASliceIndex() );

  uint16 maxIndex = static_cast< Analysis * >( analysisFilter )->GetIntensitiesSize() - 1;
  
  lowIndexSpin->setMaximum( maxIndex );
  highIndexSpin->setMaximum( maxIndex );

  if ( static_cast< Analysis * >( analysisFilter )->GetVisualizationType() == M4D::Imaging::VT_PARAM ) {
    emit ParamaterMapsSelected();
  }

  onceFinished = true;
	
  QMessageBox::information( parent, tr( "Execution finished" ), tr( "Filter finished its work" ) );
}