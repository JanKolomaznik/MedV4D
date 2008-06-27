#include "GUI/StManagerStudyListComp.h"

#include <QtGui>

#include <vector>
#include <sstream>

// DICOM includes:
#include "ExceptionBase.h"

// DICOM namespace:
using namespace M4D::Dicom;

using namespace std;


StManagerStudyListComp::StManagerStudyListComp ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                                                 QDialog *studyManagerDialog, QWidget *parent )
  : QWidget( parent ),
    vtkRenderWindowWidget( vtkRenderWindowWidget ), studyManagerDialog( studyManagerDialog )
{
  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QVBoxLayout *buttonLayout = new QVBoxLayout;

  viewButton        = createButton( tr( "&View" ),   SLOT(view()) );
  deleteButton      = createButton( tr( "&Delete" ), SLOT(del()) );
  sendButton        = createButton( tr( "S&end" ),   SLOT(send()) );
  queueFilterButton = createButton( tr( "&Queue" ),  SLOT(queue()) );

  viewButton->setEnabled( false );
  // buttons not implemented yet:
  deleteButton->setEnabled( false );
  sendButton->setEnabled( false );
  queueFilterButton->setEnabled( false );

  buttonLayout->addWidget( viewButton );
  buttonLayout->addWidget( deleteButton );
  buttonLayout->addWidget( sendButton );
  buttonLayout->addWidget( queueFilterButton );

  QSpacerItem *verticalSpacer = new QSpacerItem( 2, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Expanding );
  buttonLayout->addItem( verticalSpacer );

  // =-=-=-=-=-=-=-=- Spacer -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QSpacerItem *horizontalSpacer = new QSpacerItem( 8, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Minimum );

  // =-=-=-=-=-=-=-=- Tabs -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  
  studyListTab = new QTabWidget;
  connect( studyListTab, SIGNAL(currentChanged(int)), this, SLOT(activeTabChanged()) );
  connect( studyListTab, SIGNAL(currentChanged(int)), this, SLOT(setEnabledView()) );

  // Recent Exams tab
  QHBoxLayout *recentExamsLayout = new QHBoxLayout;
  
  recentExamsTable = createStudyTable();
  activeExamTable  = recentExamsTable;
  recentExamsLayout->addWidget( recentExamsTable );
  
  QWidget *recentExamsPane = new QWidget;
  recentExamsPane->setLayout( recentExamsLayout );
  studyListTab->addTab( recentExamsPane, QIcon( ":/icons/recent.png" ), tr( "Recent Exams" ) );

  // Remote Exams tab
  QHBoxLayout *remoteExamsLayout = new QHBoxLayout;
  
  remoteExamsTable = createStudyTable();
  remoteExamsLayout->addWidget( remoteExamsTable );
  
  QWidget *remoteExamsPane = new QWidget;
  remoteExamsPane->setLayout( remoteExamsLayout );
  studyListTab->addTab( remoteExamsPane, QIcon( ":/icons/remote.png" ), tr( "Remote Exams" ) );

  // DICOMDIR tab
  QHBoxLayout *DICOMDIRLayout = new QHBoxLayout;
  
  DICOMDIRTable = createStudyTable();
  DICOMDIRLayout->addWidget( DICOMDIRTable );

  directoryTree = new QTreeView;
  directoryTree->setFixedWidth( 280 );
  QDirModel *model = new QDirModel();
  model->setFilter( QDir::Dirs | QDir::NoDotAndDotDot | QDir::Drives );
  directoryTree->setModel( model );
  directoryTree->setColumnWidth( 0, 150 );
  // hide size and type in filesystem tree view
  directoryTree->setColumnHidden( 1, true );
  directoryTree->setColumnHidden( 2, true );
  // directoryTree->header()->hide();
  DICOMDIRLayout->addWidget( directoryTree );

  QWidget *DICOMDIRPane = new QWidget;
  DICOMDIRPane->setLayout( DICOMDIRLayout );
  studyListTab->addTab( DICOMDIRPane, QIcon( ":/icons/dicomdir.png" ), tr( "DICOMDIR" ) );

  // =-=-=-=-=-=-=-=- Study List -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QHBoxLayout *studyListLayout = new QHBoxLayout;
  studyListLayout->addLayout( buttonLayout );
  studyListLayout->addItem( horizontalSpacer );
  studyListLayout->addWidget( studyListTab );

  setLayout( studyListLayout );

  // DICOM initializations:
  dcmProvider = new DcmProvider();

  recentResultSet   = new DcmProvider::ResultSet();
  remoteResultSet   = new DcmProvider::ResultSet();
  DICOMDIRResultSet = new DcmProvider::ResultSet();
  
  activeResultSet = recentResultSet;
}


StManagerStudyListComp::~StManagerStudyListComp ()
{
  delete dcmProvider;
  delete recentResultSet;
  delete remoteResultSet;
  delete DICOMDIRResultSet;
}


void StManagerStudyListComp::find ( const string &firstName, const string &lastName,
                                    const string &patientID, 
                                    const string &fromDate, const string &toDate,
                                    const DcmProvider::StringVector &modalitiesVect )
{
  try {

    QSettings settings( "dd", "ss" );
    
    QString DICOMDIRPath;
    QModelIndex qm;

    switch ( studyListTab->currentIndex() )
    {
      case 0:
        // Recent Exams tab active
        recentResultSet->clear();

        QMessageBox::warning( this, tr( "Settings" ), settings.value( "firstName" ).toString() ); 

        if ( recentResultSet->empty() ) {
          QMessageBox::warning( this, tr( "No results" ), 
                                "Recent Exams - No search results match your criteria" );
        }
        break;

      case 1:
        // Remote Exams tab active
        remoteResultSet->clear();
        
        dcmProvider->Find( *remoteResultSet, firstName, lastName, patientID, modalitiesVect,
                            fromDate, toDate );	
 	  
        // it can handle empty resultSet
        addResultSetToStudyTable( remoteExamsTable );

        if ( remoteResultSet->empty() ) {
          QMessageBox::warning( this, tr( "No results" ), 
                                "Remote Exams - No search results match your criteria" );
        }
        break;

      case 2:
        // DICOMDIR tab active
        DICOMDIRResultSet->clear();

        if ( !directoryTree->selectionModel()->selectedIndexes().empty() )
        {
          qm = directoryTree->selectionModel()->selectedIndexes()[0];
          DICOMDIRPath = ((QDirModel *)directoryTree->model())->filePath( qm );
          QMessageBox::warning( this, tr( "Path" ), DICOMDIRPath );
        }
        else 
        {
          DICOMDIRPath = QDir::currentPath();
          QMessageBox::warning( this, tr( "Path" ), DICOMDIRPath );
        }

        dcmProvider->FindInFolder( *DICOMDIRResultSet, DICOMDIRPath.toStdString() );

        // it can handle empty resultSet
        addResultSetToStudyTable( DICOMDIRTable );
        
        if ( DICOMDIRResultSet->empty() ) {
          QMessageBox::warning( this, tr( "No results" ), 
                                "DICOMDIR - No search results match your criteria" );
        }
        break;

      default:
        recentResultSet->clear();

        if ( recentResultSet->empty() ) {
          QMessageBox::warning( this, tr( "No results" ), 
                                "Recent Exams - No search results match your criteria" );
        }
        break;
    }

  } 
  catch ( M4D::ErrorHandling::ExceptionBase & e ) {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  } 
  catch( std::exception &e ) {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  }
}


// progessBar needed...
void StManagerStudyListComp::view ()
{
  // this test is not necessary (view button is disabled when no selection)
  if ( !activeExamTable->selectedItems().empty() )
  {
    DcmProvider::StudyInfo *studyInfo     = new DcmProvider::StudyInfo();
	  DcmProvider::DicomObjSet *dicomObjSet = new DcmProvider::DicomObjSet();	

    // we are sure, there is exactly one selected
    int selectedRow = activeExamTable->selectedItems()[0]->row();
    DcmProvider::TableRow *row = &activeResultSet->at( selectedRow );

	  // find some info about selected study
	  dcmProvider->WholeFindStudyInfo( row->patentID, row->studyID, *studyInfo );

	  // now get image
	  dcmProvider->GetImageSet( row->patentID, row->studyID, studyInfo->begin()->first, *dicomObjSet );

    vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->imageDataToRenderWindow( DcmProvider::DicomObjSetPtr( dicomObjSet ) ) );

    // add to Recent Exams
    QSettings settings( "dd", "ss" );
    settings.setValue( "firstName", QString( row->patientName.c_str() ) );

    delete studyInfo;

    studyManagerDialog->close();
  }
}


void StManagerStudyListComp::setEnabledView ()
{
  !activeExamTable->selectedItems().empty() ? viewButton->setEnabled( true ) : 
                                              viewButton->setEnabled( false );
}


void StManagerStudyListComp::activeTabChanged ()
{
  switch ( studyListTab->currentIndex() )
  {
      case 0:
        // Recent Exams tab active
        activeExamTable = recentExamsTable;
        activeResultSet = recentResultSet;
        break;

      case 1:
        // Remote Exams tab active
        activeExamTable = remoteExamsTable;
        activeResultSet = remoteResultSet;
        break;

      case 2:
        // DICOMDIR tab active
        activeExamTable = DICOMDIRTable;
        activeResultSet = DICOMDIRResultSet;
        break;

      default:
        activeExamTable = recentExamsTable;
        activeResultSet = recentResultSet;
        break;
  }
}


void StManagerStudyListComp::addResultSetToStudyTable ( QTableWidget *table )
{
  // for correct inserting sorting must be disabled
  table->setSortingEnabled( false );

  table->clearContents();
  table->setRowCount( 0 );

  for ( unsigned rowNum = 0; rowNum < activeResultSet->size(); rowNum++ ) {
    addRowToStudyTable( &activeResultSet->at( rowNum ), table );
  }

  // sorting must be enabled AFTER populating table with items
  table->setSortingEnabled( true );
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
  QDate studyDate = QDate::fromString( QString( row->studyDate.c_str() ), "yyyyMMdd" );
  tableRowItems.push_back( new QTableWidgetItem( studyDate.toString( "dd. MM. yyyy" ) ) );
  // Time:
  tableRowItems.push_back( new QTableWidgetItem( QString( "" ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->studyID.c_str() ) ) );
  tableRowItems.push_back( new QTableWidgetItem( row->patientSex ? QString( tr( "male" ) ) : 
                                                                   QString( tr( "female" ) ) ) );
  QDate patientBirthDate = QDate::fromString( QString( row->patientBirthDate.c_str() ), "yyyyMMdd" );
  tableRowItems.push_back( new QTableWidgetItem( patientBirthDate.toString( "dd. MM. yyyy" ) ) );
  // And the others....

  for ( unsigned colNum = 0; colNum < tableRowItems.size(); colNum ++ ) {  
    table->setItem( rowNum, colNum, tableRowItems[colNum] );
  }
  table->setRowHeight( rowNum, 23 );
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

  connect( table, SIGNAL(itemSelectionChanged()), this, SLOT(setEnabledView()) );

  return table;
}


QPushButton *StManagerStudyListComp::createButton ( const QString &text, const char *member )
{
  QPushButton *button = new QPushButton( text );
  connect( button, SIGNAL(clicked()), this, member );

  return button;
}

