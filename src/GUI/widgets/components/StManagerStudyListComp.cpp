#include "GUI/StManagerStudyListComp.h"

#include <QtGui>

#include <vector>
#include <sstream>

// DICOM includes:
#include "Common.h"
#include "ExceptionBase.h"
#include "M4DDICOMServiceProvider.h"

// DICOM namespace:
using namespace M4D::Dicom;

using namespace std;


StManagerStudyListComp::StManagerStudyListComp ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                                                 QWidget *parent )
  : QWidget( parent ),
    vtkRenderWindowWidget( vtkRenderWindowWidget )
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


StManagerStudyListComp::~StManagerStudyListComp ()
{
  delete dcmProvider;
  delete resultSet;
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
  // this test won't be necessary (view button enabled/disabled)
  if ( !localExamsTable->selectedItems().empty() )
  {
    DcmProvider::StudyInfo *studyInfo     = new DcmProvider::StudyInfo();
	  DcmProvider::DicomObjSet *dicomObjSet = new DcmProvider::DicomObjSet();	

    // we are sure, there is exactly one selected
    int selectedRow = localExamsTable->selectedItems()[0]->row();
    DcmProvider::TableRow *row = &resultSet->at( selectedRow );

	  // find some info about selected study
	  dcmProvider->WholeFindStudyInfo( row->patentID, row->studyID, *studyInfo );

	  // now get image
	  dcmProvider->GetImageSet( row->patentID, row->studyID, studyInfo->begin()->first, *dicomObjSet );

    /*
    // just save the dcms and open them with vtkReader (like Open)...
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    stringstream saveDirectory;
    saveDirectory << QDir::current().path().toStdString() << QDir::separator().toAscii() << selectedRow;
    QDir dir( saveDirectory.str().c_str() );
    if ( !dir.exists() ) {
      if ( !dir.mkdir( dir.path() ) ) {
          QMessageBox::warning( this, "Cannot make directory", "Could not create directory " );
      }
    }

    int i = 0;
	  for ( DcmProvider::DicomObjSet::iterator it = dicomObjSet->begin(); it != dicomObjSet->end(); it++ )
	  {
      stringstream saveName;
      saveName << dir.absolutePath().toStdString() << QDir::separator().toAscii() << 
                  i++ << ".dcm";
      it->Save( saveName.str() );
	  }
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->dicomToRenderWindow( saveDirectory.str().c_str() ) );
    */
	  
    vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->imageDataToRenderWindow( DcmProvider::DicomObjSetPtr( dicomObjSet ) ) );

    delete studyInfo;
  }
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

  connect( table, SIGNAL(update()), this, SLOT(!table->selectedItems().empty() ? 
                                               viewButton->setEnabled( true ) :
                                               viewButton->setEnabled( false )) );

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