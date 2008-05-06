#include "StManagerStudyListComp.h"

#include <QtGui>

#include <vector>

using namespace std;
// DICOM namespace:
using namespace M4D::Dicom;


StManagerStudyListComp::StManagerStudyListComp ( QWidget *parent )
  : QWidget( parent )
{
  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QVBoxLayout *buttonLayout = new QVBoxLayout;

  viewButton        = createButton( tr( "&View" ),          SLOT(search()) );
  deleteButton      = createButton( tr( "&Delete" ),        SLOT(today()) );
  sendButton        = createButton( tr( "S&end" ),          SLOT(yesterday()) );
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
}


void StManagerStudyListComp::addResultSetToStudyTable ( const DcmProvider::ResultSet *resultSet )
{
  // there are more tables...we need to specify the desired one...

  for ( unsigned rowNum = 0; rowNum < resultSet->size(); rowNum++ ) {
    addRowToStudyTable( &resultSet->at( rowNum ) );
  }
}


QTableWidget *StManagerStudyListComp::createStudyTable ()
{
  QTableWidget *table  = new QTableWidget;

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


void StManagerStudyListComp::addRowToStudyTable ( const DcmProvider::TableRow *row )
{
  // there are more tables...we need to specify the desired one...
 
  //int rowNum = localExamsTable->rowCount();
  //localExamsTable->setRowCount( rowNum + 1 );

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
  tableRowItems.push_back( new QTableWidgetItem( QString( row->patientSex ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->patientBirthDate.c_str() ) ) );
  // And the others....

  for ( unsigned colNum = 0; colNum < tableRowItems.size(); colNum ++ ) {
    //localExamsTable->setItem( 0, colNum, tableRowItems[colNum] );
  }
}


QPushButton *StManagerStudyListComp::createButton ( const QString &text, const char *member )
{
  QPushButton *button = new QPushButton( text );
  connect( button, SIGNAL(clicked()), this, member );

  return button;
}