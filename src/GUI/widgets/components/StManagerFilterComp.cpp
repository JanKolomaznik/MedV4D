#include "StManagerFilterComp.h"

#include <QtGui>

// DICOM includes:
#include "Common.h"
#include "ExceptionBase.h"
#include "M4DDICOMServiceProvider.h"

// DICOM namespace:
using namespace M4D::Dicom;


StManagerFilterComp::StManagerFilterComp ( StManagerStudyListComp *studyListComponent, QWidget *parent )
  : QWidget( parent ),
    studyListComponent( studyListComponent )
{
  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QVBoxLayout *buttonLayout = new QVBoxLayout;

  searchButton      = createButton( tr( "&Search" ),       SLOT(search()) );
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

  // =-=-=-=-=-=-=-=- Spacer -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QSpacerItem *horizontalSpacer = new QSpacerItem( 8, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Minimum );

  // =-=-=-=-=-=-=-=- Inputs -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  
  QGridLayout *inputLayout = new QGridLayout;
  
  QLabel *patientIDLabel = new QLabel( tr( "Patient ID:" ) );
  QLabel *lastNameLabel  = new QLabel( tr( "Last Name:" ) );
  QLabel *firstNameLabel = new QLabel( tr( "First Name:" ) );
  
  patientIDComboBox = createComboBox();
  lastNameComboBox  = createComboBox();
  firstNameComboBox = createComboBox();

  QSpacerItem *vertRow12Spacer = new QSpacerItem( 2, 5, QSizePolicy::Minimum, 
                                                  QSizePolicy::Minimum );

  QCheckBox *fromDateCheckBox = createCheckBox( tr( "From:" ), SLOT(fromCheck()) );
  QCheckBox *toDateCheckBox   = createCheckBox( tr( "To:" ),   SLOT(toCheck())  );

  fromDateDateEdit = new QDateEdit( QDate::currentDate() );
  fromDateDateEdit->setDisplayFormat( "d. M. yyyy" );
  fromDateDateEdit->setCalendarPopup( true );
  toDateDateEdit   = new QDateEdit( QDate::currentDate() );
  toDateDateEdit->setDisplayFormat( "d. M. yyyy" );
  toDateDateEdit->setCalendarPopup( true );

  QSpacerItem *vertRow23Spacer = new QSpacerItem( 2, 6, QSizePolicy::Minimum, 
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

  inputLayout->addItem( vertRow12Spacer, 2, 0, 1, 3 );
  
  inputLayout->addWidget( fromDateCheckBox,  3, 0 );
  inputLayout->addWidget( toDateCheckBox,    3, 1 );
  inputLayout->addWidget( fromDateDateEdit,  4, 0 );
  inputLayout->addWidget( toDateDateEdit,    4, 1 );

  inputLayout->addItem( vertRow23Spacer, 5, 0, 1, 3 );
  
  inputLayout->addWidget( accesionLabel,       6, 0 );
  inputLayout->addWidget( studyDescLabel,      6, 1 );
  inputLayout->addWidget( referringMDLabel,    6, 2 );
  inputLayout->addWidget( accesionComboBox,    7, 0 );
  inputLayout->addWidget( studyDescComboBox,   7, 1 );
  inputLayout->addWidget( referringMDComboBox, 7, 2 );

  QSpacerItem *verticalSpacer = new QSpacerItem( 2, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Expanding );
  inputLayout->addItem( verticalSpacer, 8, 0, 1, 3 );

  // =-=-=-=-=-=-=-=- Filter -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QHBoxLayout *filterLayout = new QHBoxLayout;
  filterLayout->addLayout( buttonLayout );
  filterLayout->addItem( horizontalSpacer );
  filterLayout->addLayout( inputLayout );

  setLayout( filterLayout );
}


void StManagerFilterComp::search ()
{
  QString patientNameText = firstNameComboBox->currentText() + " "
                          + lastNameComboBox->currentText();
  QString patientIDText   = patientIDComboBox->currentText();
    
  QString fromDateText    = fromDateDateEdit->date().toString( "yyyyMMdd" );
  QString toDateText      = toDateDateEdit->date().toString( "yyyyMMdd" );

  DcmProvider::ResultSet *resultSet     = new DcmProvider::ResultSet();
  DcmProvider::StudyInfo *studyInfo     = new DcmProvider::StudyInfo();
	DcmProvider::DicomObjSet *dicomObjSet = new DcmProvider::DicomObjSet();	

  try {
	  DcmProvider dcmProvider;
	  DcmProvider::StringVector modalities;

    dcmProvider.Find( *resultSet, patientNameText.toStdString(), 
                      patientIDText.toStdString(), modalities,
                      fromDateText.toStdString(), toDateText.toStdString() );	

 
    std::vector<DcmProvider::TableRow> *r = new std::vector<DcmProvider::TableRow>();
   
    DcmProvider::TableRow t;
    t.patientName = "John Doe";
    t.modality = "MR";
    t.patentID = "333";
    t.patientBirthDate = "01011970";
    t.patientSex = true;
    t.studyDate = "01012008";
    t.studyID = "888";

    r->push_back( t );
   
    studyListComponent->addResultSetToStudyTable( r );
    // ---------

    if ( !resultSet->empty() )
    {
      DcmProvider::TableRow *row = &resultSet->at( 0 );

      if ( studyListComponent ) {
        studyListComponent->addResultSetToStudyTable( resultSet );
      }

      // -=-=-=-=-=- after clicking on the row.... -=-=-=-=-=-=-=-=-
        
      // find some info about selected study
		  dcmProvider.WholeFindStudyInfo( row->patentID, row->studyID, *studyInfo );

		  // now get image
		  dcmProvider.GetImageSet( row->patentID, row->studyID, 
                               studyInfo->begin()->first, *dicomObjSet );
    }
    else
    {
      QMessageBox::warning( this, tr( "No results" ), "No search results match your criteria" );
    }
  } 
  catch ( M4D::ErrorHandling::ExceptionBase & e ) 
  {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  } 
  catch( std::exception &e ) 
  {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  }
  
  QString message = "provider.Find( " + patientNameText + ", "
                   + patientIDText + ", " + fromDateText + " )";
  QMessageBox::about( this, tr( "Search" ), message );
}


void StManagerFilterComp::fromCheck ()
{ 
  if ( fromDateDateEdit->isEnabled() ) 
    fromDateDateEdit->setEnabled( false );
  else
    fromDateDateEdit->setEnabled( true );
}


void StManagerFilterComp::toCheck ()
{ 
  if ( toDateDateEdit->isEnabled() ) 
    toDateDateEdit->setEnabled( false );
  else
    toDateDateEdit->setEnabled( true );
}


void StManagerFilterComp::modalityCheck ()
{ 
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


QCheckBox *StManagerFilterComp::createCheckBox ( const QString &text, const char *member )
{
  QCheckBox *checkBox = new QCheckBox( text );
  connect( checkBox, SIGNAL(clicked()), this, member );
  checkBox->setChecked( true );

  return checkBox;
}