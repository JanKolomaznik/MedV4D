#include <QtGui>

#include "m4dGUIStudyManagerWidget.h"


m4dGUIStudyManagerWidget::m4dGUIStudyManagerWidget ( QMainWindow *parent )
  : QWidget( parent )
{
  createFilterGroupBox();
  createHangingProtocolsGroupBox();
  createStudyListGroupBox();

  QGridLayout *mainLayout = new QGridLayout;
  mainLayout->addWidget( filterGroupBox, 0, 0 );
  mainLayout->addWidget( hangingProtocolsGroupBox, 0, 1 );
  mainLayout->addWidget( studyListGroupBox, 1, 0, 1, 2 );
  setLayout( mainLayout );

  setWindowTitle( tr( "Study Manager" ) );
}


void m4dGUIStudyManagerWidget::search ()
{
  QString patientIDText = patientIDComboBox->currentText();
  // provider.Find( result, patientName, patientIDText, modality, dateFrom);
}


void m4dGUIStudyManagerWidget::createFilterGroupBox ()
{
  filterGroupBox = new QGroupBox( tr( "Filter" ) );

  // Filter GroupBox - no resize - during resizing the main widget
  filterGroupBox->setSizePolicy( QSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed ) );

  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QVBoxLayout *buttonLayout = new QVBoxLayout;

  searchButton      = createButton( tr( "Se&arch" ),       SLOT(search()) );
  todayButton       = createButton( tr( "&Today" ),        SLOT(today()) );
  yesterdayButton   = createButton( tr( "&Yesterday" ),    SLOT(yesterday()) );
  clearFilterButton = createButton( tr( "&Clear Filter" ), SLOT(clearFilter()) );
  optionsButton     = createButton( tr( "&Options" ),      SLOT(options()) );

  // buttons not implemented yet:
  todayButton->setEnabled( false );
  yesterdayButton->setEnabled( false );
  clearFilterButton->setEnabled( false );
  optionsButton->setEnabled( false );

  buttonLayout->addWidget( searchButton );
  buttonLayout->addWidget( todayButton );
  buttonLayout->addWidget( yesterdayButton );
  buttonLayout->addWidget( clearFilterButton );
  buttonLayout->addWidget( optionsButton );

  // =-=-=-=-=-=-=-=- Inputs -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  
  QGridLayout *inputLayout = new QGridLayout;
  
  patientIDComboBox = createComboBox();
  lastNameComboBox  = createComboBox();
  firstNameComboBox = createComboBox();

  QLabel *patientIDLabel = new QLabel( tr( "Patient ID:" ) );
  QLabel *lastNameLabel  = new QLabel( tr( "Last Name:" ) );
  QLabel *firstNameLabel = new QLabel( tr( "First Name:" ) );
   
  inputLayout->addWidget( patientIDLabel,    0, 0 );
  inputLayout->addWidget( lastNameLabel,     0, 1 );
  inputLayout->addWidget( firstNameLabel,    0, 2 );
  inputLayout->addWidget( patientIDComboBox, 1, 0 );
  inputLayout->addWidget( lastNameComboBox,  1, 1 );
  inputLayout->addWidget( firstNameComboBox, 1, 2 );

  QSpacerItem *verticalSpacer = new QSpacerItem( 2, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Expanding );
  inputLayout->addItem( verticalSpacer, 2, 0, 1, 2);

  // =-=-=-=-=-=-=-=- Filter -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QHBoxLayout *filterLayout = new QHBoxLayout;
  filterLayout->addLayout( buttonLayout );
  filterLayout->addLayout( inputLayout );

  filterGroupBox->setLayout( filterLayout );
}


void m4dGUIStudyManagerWidget::createHangingProtocolsGroupBox ()
{
  hangingProtocolsGroupBox = new QGroupBox( tr( "Hanging protocols" ) );
}


void m4dGUIStudyManagerWidget::createStudyListGroupBox ()
{
  studyListGroupBox = new QGroupBox( tr( "Study List" ) );

  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QVBoxLayout *buttonLayout = new QVBoxLayout;

  viewButton        = createButton( tr( "&View" ),          SLOT(search()) );
  deleteButton      = createButton( tr( "&Delete" ),        SLOT(today()) );
  sendButton        = createButton( tr( "&Send" ),          SLOT(yesterday()) );
  queueFilterButton = createButton( tr( "&Queue" ),         SLOT(clearFilter()) );
  burnToMediaButton = createButton( tr( "&Burn to Media" ), SLOT(options()) );

  // buttons not implemented yet:
  viewButton->setEnabled( false );
  deleteButton->setEnabled( false );
  sendButton->setEnabled( false );
  queueFilterButton->setEnabled( false );
  burnToMediaButton->setEnabled( false );

  buttonLayout->addWidget( viewButton );
  buttonLayout->addWidget( deleteButton );
  buttonLayout->addWidget( sendButton );
  buttonLayout->addWidget( queueFilterButton );
  buttonLayout->addWidget( burnToMediaButton );

  QSpacerItem *verticalSpacer = new QSpacerItem( 2, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Expanding );
  buttonLayout->addItem( verticalSpacer );

  // =-=-=-=-=-=-=-=- Tabs -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  
  studyListTab = new QTabWidget;

  localExamsTable  = new QTableWidget;
  remoteExamsTable = new QTableWidget;

  QStringList labels;
  labels << tr( "Patient ID" ) << tr( "Name" ) << tr( "Accesion" )
         << tr( "Modality" ) << tr( "Description" ) << tr( "Date" )
         << tr( "Time" ) << tr( "Study ID" ) << tr( "Sex" )
         << tr( "Birthdate" ) << tr( "Referring MD" ) << tr( "Institution" )
         << tr( "Location" ) << tr( "Server" );
  
  localExamsTable->setColumnCount( 14 );
  localExamsTable->setHorizontalHeaderLabels( labels );

  studyListTab->addTab( localExamsTable, QIcon( ":/icons/local.png" ), tr( "Local Exams" ) );
  studyListTab->addTab( remoteExamsTable,  QIcon( ":/icons/remote.png" ), tr( "Remote Exams" ) );

  // =-=-=-=-=-=-=-=- Study List -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QHBoxLayout *studyListLayout = new QHBoxLayout;
  studyListLayout->addLayout( buttonLayout );
  studyListLayout->addWidget( studyListTab );

  studyListGroupBox->setLayout( studyListLayout ); 
}


QPushButton *m4dGUIStudyManagerWidget::createButton( const QString &text, const char *member )
{
  QPushButton *button = new QPushButton( text );
  connect( button, SIGNAL(clicked()), this, member );

  return button;
}


QComboBox *m4dGUIStudyManagerWidget::createComboBox ( const QString &text )
{
  QComboBox *comboBox = new QComboBox;
  comboBox->setEditable( true );
  comboBox->addItem( text );
  comboBox->setSizePolicy( QSizePolicy::Expanding, QSizePolicy::Preferred );
  comboBox->setMinimumWidth( 100 );
    
  return comboBox;
 }