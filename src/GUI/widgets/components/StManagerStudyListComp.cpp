#include "GUI/StManagerStudyListComp.h"

#include <QtGui>

#include <vector>

// DICOM includes:
#include "Common.h"
#include "ExceptionBase.h"
#include "M4DDICOMServiceProvider.h"

// DICOM namespace:
using namespace M4D::Dicom;

using namespace std;


StManagerStudyListComp::StManagerStudyListComp ( QWidget *parent )
  : QWidget( parent )
{
  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QVBoxLayout *buttonLayout = new QVBoxLayout;

  viewButton        = createButton( tr( "&View" ),          SLOT(view()) );
  deleteButton      = createButton( tr( "&Delete" ),        SLOT(del()) );
  sendButton        = createButton( tr( "S&end" ),          SLOT(send()) );
  queueFilterButton = createButton( tr( "&Queue" ),         SLOT(queue()) );
  burnToMediaButton = createButton( tr( "&Burn to Media" ), SLOT(burn()) );

  // viewButton->setEnabled( false );
  // buttons not implemented yet:
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

  // =-=-=-=-=-=-=-=- Spacer -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QSpacerItem *horizontalSpacer = new QSpacerItem( 8, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Minimum );

  // =-=-=-=-=-=-=-=- Tabs -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  
  studyListTab = new QTabWidget;

  localExamsTable  = createStudyTable();
  remoteExamsTable = createStudyTable();

  studyListTab->addTab( localExamsTable, QIcon( ":/icons/local.png" ), tr( "Local Exams" ) );
  studyListTab->addTab( remoteExamsTable, QIcon( ":/icons/remote.png" ), tr( "Remote Exams" ) );

  // =-=-=-=-=-=-=-=- Study List -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QHBoxLayout *studyListLayout = new QHBoxLayout;
  studyListLayout->addLayout( buttonLayout );
  studyListLayout->addItem( horizontalSpacer );
  studyListLayout->addWidget( studyListTab );

  setLayout( studyListLayout );

  // DICOM initializations:
  dcmProvider = new DcmProvider();
  resultSet   = new DcmProvider::ResultSet();
}


void StManagerStudyListComp::find ( const QString &patientName, const QString &patientID, 
                                    const QString &fromDate, const QString &toDate )
{
  try {
	  DcmProvider::StringVector modalities;

    dcmProvider->Find( *resultSet, patientName.toStdString(), 
                        patientID.toStdString(), modalities,
                        fromDate.toStdString(), toDate.toStdString() );	
 	
    if ( !resultSet->empty() ) {
      addResultSetToStudyTable( localExamsTable );
    }
    else {
      QMessageBox::warning( this, tr( "No results" ), "No search results match your criteria" );
    }
  } 
  catch ( M4D::ErrorHandling::ExceptionBase & e ) {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  } 
  catch( std::exception &e ) {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  }
}


void StManagerStudyListComp::addResultSetToStudyTable ( QTableWidget *table )
{
  table->clearContents();
  table->setRowCount( 0 );

  for ( unsigned rowNum = 0; rowNum < resultSet->size(); rowNum++ ) {
    addRowToStudyTable( &resultSet->at( rowNum ), table );
  }
}


void StManagerStudyListComp::view ()
{
  /*
  DcmProvider::StudyInfo *studyInfo     = new DcmProvider::StudyInfo();
	DcmProvider::DicomObjSet *dicomObjSet = new DcmProvider::DicomObjSet();	

  DcmProvider::TableRow *row = &result[0];

	// find some info about selected study
	provider.WholeFindStudyInfo( row->patentID, row->studyID, studyInfo);

	// now get image
	provider.GetImageSet( row->patentID, row->studyID,
	studyInfo.begin()->first, obj);
  */
}


QTableWidget *StManagerStudyListComp::createStudyTable ()
{
  QTableWidget *table  = new QTableWidget;

  table->setSelectionBehavior( QAbstractItemView::SelectRows );
  table->setSelectionMode( QAbstractItemView::SingleSelection );
  table->setEditTriggers( QAbstractItemView::NoEditTriggers );

  QStringList labels;
  labels << tr( "Patient ID" ) << tr( "Name" ) << tr( "Accesion" )
         << tr( "Modality" ) << tr( "Description" ) << tr( "Date" )
         << tr( "Time" ) << tr( "Study ID" ) << tr( "Sex" )
         << tr( "Birthdate" ) << tr( "Referring MD" ) << tr( "Institution" )
         << tr( "Location" ) << tr( "Server" );
  
  table->setColumnCount( labels.size() );
  table->setHorizontalHeaderLabels( labels );

  return table;
}


void StManagerStudyListComp::addRowToStudyTable ( const DcmProvider::TableRow *row, 
                                                  QTableWidget *table )
{
  int rowNum = table->rowCount();
  table->setRowCount( rowNum + 1 );

  vector< QTableWidgetItem * > tableRowItems;
  tableRowItems.push_back( new QTableWidgetItem( QString( row->patentID.c_str() ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->patientName.c_str() ) ) );
  // Accesion:
  tableRowItems.push_back( new QTableWidgetItem( QString( "" ) ) ); 
  tableRowItems.push_back( new QTableWidgetItem( QString( row->modality.c_str() ) ) );
  // Description:
  tableRowItems.push_back( new QTableWidgetItem( QString( "" ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->studyDate.c_str() ) ) );
  // Time:
  tableRowItems.push_back( new QTableWidgetItem( QString( "" ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->studyID.c_str() ) ) );
  tableRowItems.push_back( new QTableWidgetItem( row->patientSex ? QString( tr( "male" ) ) : 
                                                                   QString( tr( "female" ) ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->patientBirthDate.c_str() ) ) );
  // And the others....

  for ( unsigned colNum = 0; colNum < tableRowItems.size(); colNum ++ ) {  
    table->setItem( rowNum, colNum, tableRowItems[colNum] );
  }
}


QPushButton *StManagerStudyListComp::createButton ( const QString &text, const char *member )
{
  QPushButton *button = new QPushButton( text );
  connect( button, SIGNAL(clicked()), this, member );

  return button;
}