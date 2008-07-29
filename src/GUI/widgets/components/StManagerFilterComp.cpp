#include "GUI/StManagerFilterComp.h"

#include <QtGui>

#include <vector>
#include <string>

using namespace std;


namespace M4D {
namespace GUI {

/// Number of possible modalities - in filter - number of checkboxes
#define MODALITY_NUMBER   14

const char *StManagerFilterComp::modalities[] = { "CR", "ES", "NM", "RF", "US", "CT", "MG", 
                                                  "OT", "RT", "XA", "DX", "MR", "PT", "SC" };

StManagerFilterComp::StManagerFilterComp ( StManagerStudyListComp *studyListComponent, QWidget *parent )
  : QWidget( parent ),
    studyListComponent( studyListComponent )
{
  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QVBoxLayout *buttonLayout = new QVBoxLayout;

  searchButton      = createButton( tr( "&Search" ),       SLOT(search()) );
  todayButton       = createButton( tr( "&Today" ),        SLOT(today()) );
  yesterdayButton   = createButton( tr( "&Yesterday" ),    SLOT(yesterday()) );
  clearFilterButton = createButton( tr( "&Clear Filter" ), SLOT(clear()) );
  optionsButton     = createButton( tr( "&Options" ),      SLOT(options()) );

  // button not implemented yet:
  optionsButton->setEnabled( false );

  buttonLayout->addWidget( searchButton );
  buttonLayout->addWidget( todayButton );
  buttonLayout->addWidget( yesterdayButton );
  buttonLayout->addWidget( clearFilterButton );
  buttonLayout->addWidget( optionsButton );

  // =-=-=-=-=-=-=-=- Spacer -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QSpacerItem *horSpacerButInp = new QSpacerItem( 8, 2, QSizePolicy::Minimum, 
                                                  QSizePolicy::Minimum );

  // =-=-=-=-=-=-=-=- Inputs -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  
  QGridLayout *inputLayout = new QGridLayout;
  
  QLabel *patientIDLabel = new QLabel( tr( "Patient ID:" ) );
  QLabel *lastNameLabel  = new QLabel( tr( "Last Name:" ) );
  QLabel *firstNameLabel = new QLabel( tr( "First Name:" ) );
  
  patientIDComboBox = createComboBox();
  lastNameComboBox  = createComboBox();
  firstNameComboBox = createComboBox();

  QSpacerItem *vertSpacerR1R2 = new QSpacerItem( 2, 5, QSizePolicy::Minimum, 
                                                 QSizePolicy::Minimum );

  fromDateCheckBox = createCheckBox( tr( "From:" ), false, SLOT(from()) );
  toDateCheckBox   = createCheckBox( tr( "To:" ), false, SLOT(to()) );

  fromDateDateEdit = new QDateEdit( QDate::currentDate() );
  fromDateDateEdit->setDisplayFormat( "d. M. yyyy" );
  fromDateDateEdit->setCalendarPopup( true );
  fromDateDateEdit->setEnabled( false );

  toDateDateEdit = new QDateEdit( QDate::currentDate() );
  toDateDateEdit->setDisplayFormat( "d. M. yyyy" );
  toDateDateEdit->setCalendarPopup( true );
  toDateDateEdit->setEnabled( false );

  QSpacerItem *vertSpacerR2R3 = new QSpacerItem( 2, 6, QSizePolicy::Minimum, 
                                                 QSizePolicy::Minimum );

  QLabel *accesionLabel    = new QLabel( tr( "Accession#:" ) );
  QLabel *studyDescLabel   = new QLabel( tr( "Study Desc.:" ) );
  QLabel *referringMDLabel = new QLabel( tr( "Referring MD:" ) );

  accesionComboBox    = createComboBox();
  studyDescComboBox   = createComboBox();
  referringMDComboBox = createComboBox();
   
  inputLayout->addWidget( patientIDLabel,    0, 0 );
  inputLayout->addWidget( lastNameLabel,     0, 1 );
  inputLayout->addWidget( firstNameLabel,    0, 2 );
  inputLayout->addWidget( patientIDComboBox, 1, 0 );
  inputLayout->addWidget( lastNameComboBox,  1, 1 );
  inputLayout->addWidget( firstNameComboBox, 1, 2 );

  inputLayout->addItem( vertSpacerR1R2, 2, 0, 1, 3 );
  
  inputLayout->addWidget( fromDateCheckBox,  3, 0 );
  inputLayout->addWidget( toDateCheckBox,    3, 1 );
  inputLayout->addWidget( fromDateDateEdit,  4, 0 );
  inputLayout->addWidget( toDateDateEdit,    4, 1 );

  inputLayout->addItem( vertSpacerR2R3, 5, 0, 1, 3 );
  
  inputLayout->addWidget( accesionLabel,       6, 0 );
  inputLayout->addWidget( studyDescLabel,      6, 1 );
  inputLayout->addWidget( referringMDLabel,    6, 2 );
  inputLayout->addWidget( accesionComboBox,    7, 0 );
  inputLayout->addWidget( studyDescComboBox,   7, 1 );
  inputLayout->addWidget( referringMDComboBox, 7, 2 );

  QSpacerItem *vertSpacerBottom = new QSpacerItem( 2, 2, QSizePolicy::Minimum, 
                                                   QSizePolicy::Expanding );
  inputLayout->addItem( vertSpacerBottom, 8, 0, 1, 3 );

  // =-=-=-=-=-=-=-=- Spacer -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QSpacerItem *horSpacerInpMod = new QSpacerItem( 8, 2, QSizePolicy::Minimum, 
                                                  QSizePolicy::Minimum );

  // =-=-=-=-=-=-=-=- Modalities -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  modalitiesGroupBox = new QGroupBox( tr( "Modality" ) );

  QVBoxLayout *modalitiesLayout = new QVBoxLayout;

  allCheckBox = createCheckBox( tr( "All" ), true, SLOT(all()) );

  // Spacer between All checkBox and other checkBoxes
  QSpacerItem *vertSpacerAllMod = new QSpacerItem( 2, 20, QSizePolicy::Minimum, 
                                                   QSizePolicy::Minimum );

  QGridLayout *gridModalLayout = new QGridLayout;

  modalityCheckBoxes = new QCheckBox *[MODALITY_NUMBER];

  for ( int i = 0; i < MODALITY_NUMBER; i++ )
  {
    modalityCheckBoxes[i] = createCheckBox( tr( modalities[i] ), true, SLOT(modality()) );
    gridModalLayout->addWidget( modalityCheckBoxes[i], i / 5, i % 5 );
  }

  modalitiesLayout->addWidget( allCheckBox );
  modalitiesLayout->addItem( vertSpacerAllMod );
  modalitiesLayout->addLayout( gridModalLayout );

  modalitiesGroupBox->setLayout( modalitiesLayout );

  // =-=-=-=-=-=-=-=- Filter -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QHBoxLayout *filterLayout = new QHBoxLayout;
  filterLayout->addLayout( buttonLayout );
  filterLayout->addItem( horSpacerButInp );
  filterLayout->addLayout( inputLayout );
  filterLayout->addItem( horSpacerInpMod );
  filterLayout->addWidget( modalitiesGroupBox );

  setLayout( filterLayout );
}


void StManagerFilterComp::search ()
{
  QString firstNameText = firstNameComboBox->currentText();
  QString lastNameText  = lastNameComboBox->currentText();

  QString patientIDText = patientIDComboBox->currentText();
    
  QString fromDateText  = fromDateDateEdit->isEnabled() ?
                            fromDateDateEdit->date().toString( "yyyyMMdd" ) : "";
  QString toDateText    = toDateDateEdit->isEnabled() ?
                            toDateDateEdit->date().toString( "yyyyMMdd" ) : "";
  
  // construct the modalities vector (from checked ones)
  vector< string > modalitiesVect;
  for ( int i = 0; i < MODALITY_NUMBER; i++ ) 
  {
    if ( modalityCheckBoxes[i]->isChecked() ) {
      modalitiesVect.push_back( StManagerFilterComp::modalities[i] );
    }
  }
 
  QString referringMD = referringMDComboBox->currentText();
  QString description = studyDescComboBox->currentText();

  studyListComponent->find( firstNameText.toStdString(), lastNameText.toStdString(),
                            patientIDText.toStdString(), fromDateText.toStdString(),
                            toDateText.toStdString(), modalitiesVect, referringMD.toStdString(), 
                            description.toStdString() );	
}


/**
 * today slot - for setting dateCombos to actual date by clicking on 'Today' button.
 */
void StManagerFilterComp::today ()
{
  // 'Today' clicked -> check 'from' dateCheckBox & enable 'from' dateEdit
  fromDateDateEdit->setEnabled( true );
  fromDateCheckBox->setChecked( true );
  // 'Today' clicked -> check 'to' dateCheckBox & enable 'to' dateEdit
  toDateDateEdit->setEnabled( true );
  toDateCheckBox->setChecked( true );

  QDate todayDate = QDate::currentDate();

  fromDateDateEdit->setDate( todayDate );
  toDateDateEdit->setDate( todayDate );
}


/**
 * yesterday slot - for setting dateCombos to yesterday date by clicking on 'Yesterday' button.
 */
void StManagerFilterComp::yesterday ()
{
  // 'Yesterday' clicked -> check 'from' dateCheckBox & enable 'from' dateEdit
  fromDateDateEdit->setEnabled( true );
  fromDateCheckBox->setChecked( true );
  // 'Yesterday' clicked -> check 'to' dateCheckBox & enable 'to' dateEdit
  toDateDateEdit->setEnabled( true );
  toDateCheckBox->setChecked( true );

  // substracting 1 day -> yesterday
  QDate yesterdayDate = QDate::currentDate().addDays( -1 );

  fromDateDateEdit->setDate( yesterdayDate );
  toDateDateEdit->setDate( yesterdayDate );
}


/**
 * clear slot - for clearing inputs, checkBoxes, dateCombos - filtering settings.
 */
void StManagerFilterComp::clear ()
{
  patientIDComboBox->setEditText( "" );
  lastNameComboBox->setEditText( "" );
  firstNameComboBox->setEditText( "" );

  if ( fromDateCheckBox->isChecked() ) {
    fromDateCheckBox->click();
  }
  if ( toDateCheckBox->isChecked() ) {
    toDateCheckBox->click();
  }

  accesionComboBox->setEditText( "" );
  studyDescComboBox->setEditText( "" );
  referringMDComboBox->setEditText( "" );

  allCheckBox->setEnabled( true );
  allCheckBox->setChecked( true );
  for ( int i = 0; i < MODALITY_NUMBER; i++ ) {
    modalityCheckBoxes[i]->setChecked( true );
  }
}



void StManagerFilterComp::from ()
{ 
  fromDateDateEdit->setEnabled( !fromDateDateEdit->isEnabled() );
}


void StManagerFilterComp::to ()
{ 
  toDateDateEdit->setEnabled( !toDateDateEdit->isEnabled() );
}


void StManagerFilterComp::all ()
{ 
  for ( int i = 0; i < MODALITY_NUMBER; i++ ) {
    modalityCheckBoxes[i]->setChecked( allCheckBox->isChecked() );
  }
}


void StManagerFilterComp::modality ()
{ 
  int checkedNum = 0;

  for ( int i = 0; i < MODALITY_NUMBER; i++ ) 
  {
    if ( modalityCheckBoxes[i]->isChecked() ) {
      checkedNum++;
    }
  }

  if ( checkedNum == MODALITY_NUMBER || checkedNum == 0 ) 
  {
    allCheckBox->setChecked( checkedNum );
    allCheckBox->setEnabled( true );
  }
  else
  {
    allCheckBox->setChecked( true );
    allCheckBox->setEnabled( false );
  }
}


QPushButton *StManagerFilterComp::createButton ( const QString &text, const char *member )
{
  QPushButton *button = new QPushButton( text );
  connect( button, SIGNAL(clicked()), this, member );

  return button;
}


QComboBox *StManagerFilterComp::createComboBox ( const QString &text )
{
  QComboBox *comboBox = new QComboBox;
  comboBox->setEditable( true );
  comboBox->addItem( text );
  comboBox->setSizePolicy( QSizePolicy::Expanding, QSizePolicy::Preferred );
  comboBox->setMinimumWidth( 100 );
    
  return comboBox;
}


QCheckBox *StManagerFilterComp::createCheckBox ( const QString &text, bool value, const char *member )
{
  QCheckBox *checkBox = new QCheckBox( text );
  connect( checkBox, SIGNAL(clicked()), this, member );
  checkBox->setChecked( value );

  return checkBox;
}

} // namespace GUI
} // namespace M4D
