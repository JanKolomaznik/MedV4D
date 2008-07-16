#include "GUI/StManagerStudyListComp.h"

#include <QtGui>

#include <vector>

// DICOM includes:
#include "ExceptionBase.h"

// DICOM namespace:
using namespace M4D::Dicom;

using namespace std;


#define RECENT_REMOTE_EXAMS_NAME            "Recent Remote Exams"
#define RECENT_DICOMDIR_NAME                "Recent DICOMDIR"
#define REMOTE_EXAMS_NAME                   "Remote Exams"
#define DICOMDIR_NAME                       "DICOMDIR"

/// Number of exam/image attributes (e.g in study tables)
#define ATTRIBUTE_NUMBER   17
/// Names of exam/image attributes (e.g in study tables)
const char *StManagerStudyListComp::attributeNames[] = { "Patient ID", "Name", "Accesion", "Modality",
                                                         "Description", "Date", "Time", "Study ID", "Sex",
                                                         "Birthdate", "Referring MD", "Institution",
                                                         "Location", "Server", "Availability", "Status",
                                                         "User" };
/// Name of the array in QSettings - for saving recent remote exams
#define RECENT_REMOTE_EXAMS_SETTINGS_NAME   "recentRemoteExams"
/// Name of the array in QSettings - for saving recent DICOMDIR
#define RECENT_DICOMDIR_SETTINGS_NAME       "recentDICOMDIR"
/// Number of recent exams to remember
#define RECENT_EXAMS_NUMBER                 20

StManagerStudyListComp::StManagerStudyListComp ( QDialog *studyManagerDialog, QWidget *parent )
  : QWidget( parent ),
    studyManagerDialog( studyManagerDialog )
{
  // =-=-=-=-=-=-=-=- Buttons -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  QGridLayout *buttonLayout = new QGridLayout;

  viewButton = createButton( tr( "&View" ), SLOT(view()) );
  viewButton->setEnabled( false );
  buttonLayout->addWidget( viewButton, 0, 0, 1, 2 );

  QSpacerItem *horSpacerViewOther = new QSpacerItem( 2, 28, QSizePolicy::Minimum, 
                                                     QSizePolicy::Minimum );
  buttonLayout->addItem( horSpacerViewOther, 1, 0, 1, 2 );

  pathButton = createButton( tr( "&Path" ), SLOT(path()) );
  pathButton->hide();
  buttonLayout->addWidget( pathButton, 2, 0, 1, 2 );

  recentRemoteButton = createToolButton( QIcon( ":/icons/remote.png" ), SLOT(recentChanged()) );
  recentRemoteButton->setChecked( true );
  buttonLayout->addWidget( recentRemoteButton, 3, 0 );

  recentDICOMDIRButton = createToolButton( QIcon( ":/icons/dicomdir.png" ), SLOT(recentChanged()) );
  buttonLayout->addWidget( recentDICOMDIRButton, 3, 1 );

  QSpacerItem *verticalSpacer = new QSpacerItem( 2, 2, QSizePolicy::Minimum, 
                                                 QSizePolicy::Expanding );
  buttonLayout->addItem( verticalSpacer, 4, 0, 1, 2 );

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
  
  QSplitter *DICOMDIRsplitter = new QSplitter();

  DICOMDIRTable = createStudyTable();
  DICOMDIRsplitter->addWidget( DICOMDIRTable );

  directoryTree = createDirectoryTreeView();
  DICOMDIRsplitter->addWidget( directoryTree );

  DICOMDIRLayout->addWidget( DICOMDIRsplitter );

  QWidget *DICOMDIRPane = new QWidget;
  DICOMDIRPane->setLayout( DICOMDIRLayout );
  studyListTab->addTab( DICOMDIRPane, QIcon( ":/icons/dicomdir.png" ), tr( "DICOMDIR" ) );

  // =-=-=-=-=-=-=-=- Study List -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  QHBoxLayout *studyListLayout = new QHBoxLayout;
  studyListLayout->addLayout( buttonLayout );
  QSpacerItem *horizontalSpacer = new QSpacerItem( 8, 2, QSizePolicy::Minimum, 
                                                   QSizePolicy::Minimum );
  studyListLayout->addItem( horizontalSpacer );
  studyListLayout->addWidget( studyListTab );

  setLayout( studyListLayout );

  studyManagerDialogTitle = QString( studyManagerDialog->windowTitle() );
  studyManagerDialog->setWindowTitle( studyManagerDialogTitle + QString( " - " ) +                                      
                                      QString( "0" ) + tr( " studies found on " ) +
                                      tr( RECENT_REMOTE_EXAMS_NAME ) );

  // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

    // for recent exams
    QSettings settings;
    
    // for DICOMDIR
    QString DICOMDIRPath;
    QModelIndex qm;

    const char *dialogTitle = RECENT_REMOTE_EXAMS_NAME;

    activeResultSet->clear();

    switch ( studyListTab->currentIndex() )
    {
      case 0:
        // Recent Exams tab active
        if ( recentRemoteButton->isChecked() ) {
          loadRecentExams( *activeResultSet, RECENT_REMOTE_EXAMS_SETTINGS_NAME );
        } 
        else 
        {
          loadRecentExams( *activeResultSet, RECENT_DICOMDIR_SETTINGS_NAME );

          dialogTitle = RECENT_DICOMDIR_NAME;
        }
        reverse( activeResultSet->begin(), activeResultSet->end() );
        break;

      case 1:
        // Remote Exams tab active
        dcmProvider->Find( *activeResultSet, firstName, lastName, patientID, modalitiesVect,
                            fromDate, toDate );	

        dialogTitle = REMOTE_EXAMS_NAME;
        break;

      case 2:
        // DICOMDIR tab active
        if ( !directoryTree->selectionModel()->selectedIndexes().empty() )
        {
          qm = directoryTree->selectionModel()->selectedIndexes()[0];
          DICOMDIRPath = ((QDirModel *)directoryTree->model())->filePath( qm );
        }
        else {
          DICOMDIRPath = QDir::currentPath();
        }

        dcmProvider->LocalFind( *activeResultSet, DICOMDIRPath.toStdString() );

        dialogTitle = DICOMDIR_NAME;
        break;

      default:
        if ( recentRemoteButton->isChecked() ) {
          loadRecentExams( *activeResultSet, RECENT_REMOTE_EXAMS_SETTINGS_NAME );
        }
        else
        {
          loadRecentExams( *activeResultSet, RECENT_DICOMDIR_SETTINGS_NAME );

          dialogTitle = RECENT_DICOMDIR_NAME;
        }
        reverse( activeResultSet->begin(), activeResultSet->end() );

        break;
    }

    QString resNum;
    studyManagerDialog->setWindowTitle( studyManagerDialogTitle + QString( " - " ) +                                      
                                        resNum.setNum( activeResultSet->size() ) + 
                                        tr( " studies found on " ) + tr( dialogTitle ) );

    // it can handle empty resultSet
    addResultSetToStudyTable( activeResultSet, activeExamTable );

    if ( activeResultSet->empty() ) {
      QMessageBox::warning( this, tr( "No results" ), "No search results match your criteria" );
    }

  } 
  catch ( M4D::ErrorHandling::ExceptionBase &e ) {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  } 
  catch( std::exception &e ) {
	  QMessageBox::critical( this, tr( "Exception" ), e.what() );
  }
}


void StManagerStudyListComp::view ()
{
  // no selection (not necessary test, view button is disabled) or no allocated space for dicomObjSet
  if ( activeExamTable->selectedItems().empty() || 
       sizeof( *dicomObjectSet ) != sizeof( DcmProvider::DicomObjSet ) ) {
    return;
  }

  DcmProvider::SerieInfoVector info;

  // we are sure, there is exactly one selected
  int selectedRow = activeExamTable->selectedItems()[0]->row();
  int idx = activeExamTable->item( selectedRow, ATTRIBUTE_NUMBER )->text().toInt();
  DcmProvider::TableRow *row = &activeResultSet->at( idx );

  const char *recentTypePrefix = RECENT_REMOTE_EXAMS_SETTINGS_NAME;

  // different FindStudyInfo and GetImageSet calls
  switch ( studyListTab->currentIndex() )
  {
    case 0:
      // Recent Exams tab active
      if ( recentRemoteButton->isChecked() )
      {
        // find some info about selected study
        dcmProvider->FindStudyInfo( row->patientID, row->studyID, info );

        // if( studyInfo.size() > 1) showSomeChoosingDialog()
        // now get image
        dcmProvider->GetImageSet( row->patientID, row->studyID, info[0].id, *dicomObjectSet );  
      }
      else
      {
        // find some info about selected study
        dcmProvider->LocalFindStudyInfo( row->patientID, row->studyID, info );

        // if( studyInfo.size() > 1) showSomeChoosingDialog()
        // now get image
        dcmProvider->LocalGetImageSet( row->patientID, row->studyID, info[0].id, *dicomObjectSet );

        recentTypePrefix = RECENT_DICOMDIR_SETTINGS_NAME;
      }
      break;

    case 1:
      // Remote Exams tab active
      // find some info about selected study
      dcmProvider->FindStudyInfo( row->patientID, row->studyID, info );

      // if( studyInfo.size() > 1) showSomeChoosingDialog()
      // now get image
      dcmProvider->GetImageSet( row->patientID, row->studyID, info[0].id, *dicomObjectSet );
      break;

    case 2:
      // DICOMDIR tab active
      // find some info about selected study
      dcmProvider->LocalFindStudyInfo( row->patientID, row->studyID, info );

      // if( studyInfo.size() > 1) showSomeChoosingDialog()
      // now get image
      dcmProvider->LocalGetImageSet( row->patientID, row->studyID, info[0].id, *dicomObjectSet );

      recentTypePrefix = RECENT_DICOMDIR_SETTINGS_NAME;
      break;
      
    default:
      if ( recentRemoteButton->isChecked() )
      {
        // find some info about selected study
        dcmProvider->FindStudyInfo( row->patientID, row->studyID, info );

        // if( studyInfo.size() > 1) showSomeChoosingDialog()
        // now get image
        dcmProvider->GetImageSet( row->patientID, row->studyID, info[0].id, *dicomObjectSet );  
      }
      else
      {
        // find some info about selected study
        dcmProvider->LocalFindStudyInfo( row->patientID, row->studyID, info );

        // if( studyInfo.size() > 1) showSomeChoosingDialog()
        // now get image
        dcmProvider->LocalGetImageSet( row->patientID, row->studyID, info[0].id, *dicomObjectSet );

        recentTypePrefix = RECENT_DICOMDIR_SETTINGS_NAME;
      }
      break;
  }

  // add to Recent Exams
  updateRecentExams( row, recentTypePrefix );

  studyManagerDialog->accept();
}


void StManagerStudyListComp::setEnabledView ()
{
  !activeExamTable->selectedItems().empty() ? viewButton->setEnabled( true ) : 
                                              viewButton->setEnabled( false );
}


void StManagerStudyListComp::activeTabChanged ()
{
  // shown only in DICOMDIR mode
  pathButton->hide();
  // shown only in recent mode
  recentRemoteButton->hide();
  recentDICOMDIRButton->hide();

  const char *dialogTitle = RECENT_REMOTE_EXAMS_NAME;

  switch ( studyListTab->currentIndex() )
  {
      case 0:
        // Recent Exams tab active
        activeExamTable = recentExamsTable;
        activeResultSet = recentResultSet;

        recentRemoteButton->show();
        recentDICOMDIRButton->show();

        if ( !recentRemoteButton->isChecked() ) {
          dialogTitle = RECENT_DICOMDIR_NAME;
        }
        break;

      case 1:
        // Remote Exams tab active
        activeExamTable = remoteExamsTable;
        activeResultSet = remoteResultSet;

        dialogTitle = REMOTE_EXAMS_NAME;
        break;

      case 2:
        // DICOMDIR tab active
        activeExamTable = DICOMDIRTable;
        activeResultSet = DICOMDIRResultSet;

        pathButton->show();

        dialogTitle = DICOMDIR_NAME;
        break;

      default:
        activeExamTable = recentExamsTable;
        activeResultSet = recentResultSet;

        recentRemoteButton->show();
        recentDICOMDIRButton->show();

        if ( !recentRemoteButton->isChecked() ) {
          dialogTitle = RECENT_DICOMDIR_NAME;
        }
        break;
  }

  QString resNum;
  studyManagerDialog->setWindowTitle( studyManagerDialogTitle + QString( " - " ) +                                      
                                      resNum.setNum( activeResultSet->size() ) + 
                                      tr( " studies found on " ) + tr( dialogTitle ) );
}


void StManagerStudyListComp::recentChanged ()
{
  activeResultSet->clear();
  // clear the table, reinitialize variables
  addResultSetToStudyTable( activeResultSet, activeExamTable );

  const char *dialogTitle = ( recentRemoteButton->isChecked() ? RECENT_REMOTE_EXAMS_NAME : RECENT_DICOMDIR_NAME );
  studyManagerDialog->setWindowTitle( studyManagerDialogTitle + QString( " - " ) + QString( "0" ) +                                    
                                      tr( " studies found on " ) + tr( dialogTitle ) );
}


void StManagerStudyListComp::path ()
{
  directoryTree->isHidden() ? directoryTree->show() : directoryTree->hide();
}


void StManagerStudyListComp::addResultSetToStudyTable ( const DcmProvider::ResultSet *resultSet,
                                                        QTableWidget *table )
{
  // for correct inserting sorting must be disabled
  table->setSortingEnabled( false );

  table->clearContents();
  table->setRowCount( 0 );

  for ( unsigned rowNum = 0; rowNum < activeResultSet->size(); rowNum++ ) {
    addRowToStudyTable( &resultSet->at( rowNum ), table );
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
  tableRowItems.push_back( new QTableWidgetItem( QString( row->patientID.c_str() ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->name.c_str() ) ) );
  // Accesion:
  tableRowItems.push_back( new QTableWidgetItem( QString( "" ) ) ); 
  tableRowItems.push_back( new QTableWidgetItem( QString( row->modality.c_str() ) ) );
  // Description:
  tableRowItems.push_back( new QTableWidgetItem( QString( "" ) ) );
  QDate studyDate = QDate::fromString( QString( row->date.c_str() ), "yyyyMMdd" );
  tableRowItems.push_back( new QTableWidgetItem( studyDate.toString( "dd. MM. yyyy" ) ) );
  // Time:
  tableRowItems.push_back( new QTableWidgetItem( QString( "" ) ) );
  tableRowItems.push_back( new QTableWidgetItem( QString( row->studyID.c_str() ) ) );
  tableRowItems.push_back( new QTableWidgetItem( row->sex ? QString( tr( "male" ) ) : 
                                                                   QString( tr( "female" ) ) ) );
  QDate patientBirthDate = QDate::fromString( QString( row->birthDate.c_str() ), "yyyyMMdd" );
  tableRowItems.push_back( new QTableWidgetItem( patientBirthDate.toString( "dd. MM. yyyy" ) ) );
  // And the others....

  for ( unsigned colNum = 0; colNum < tableRowItems.size(); colNum ++ ) {  
    table->setItem( rowNum, colNum, tableRowItems[colNum] );
  }

  // save original position to hidden column
  QString idx;
  table->setItem( rowNum, ATTRIBUTE_NUMBER, new QTableWidgetItem( idx.setNum( rowNum ) ) );

  table->setRowHeight( rowNum, 23 );
}


void StManagerStudyListComp::updateRecentExams ( const DcmProvider::TableRow *row, const QString &prefix )
{
  DcmProvider::ResultSet resultSet;
  loadRecentExams( resultSet, prefix );

  resultSet.push_back( *row ); 
  if ( resultSet.size() > RECENT_EXAMS_NUMBER ) {
    resultSet.erase( resultSet.begin() );
  }

  QSettings settings;
  settings.beginWriteArray( prefix );
  
  for ( int i = 0; i < resultSet.size(); i++ )
  {
    settings.setArrayIndex( i );
    updateRecentRow ( &resultSet[i], settings );  
  }

  settings.endArray();
}


void StManagerStudyListComp::loadRecentExams ( DcmProvider::ResultSet &resultSet, const QString &prefix )
{
  QSettings settings;
  int size = settings.beginReadArray( prefix );
 
  for ( int i = 0; i < size; i++ )
  {
    DcmProvider::TableRow row;

    settings.setArrayIndex( i );
    loadRecentRow( row, settings );
    
    resultSet.push_back( row );
  }

  settings.endArray();
}


void StManagerStudyListComp::updateRecentRow ( const DcmProvider::TableRow *row, QSettings &settings )
{
  // some are missing....
  settings.setValue( attributeNames[0], row->patientID.c_str() );
  settings.setValue( attributeNames[1], row->name.c_str() );
  settings.setValue( attributeNames[3], row->modality.c_str() );
  settings.setValue( attributeNames[5], row->date.c_str() );
  settings.setValue( attributeNames[7], row->studyID.c_str() );
  settings.setValue( attributeNames[8], row->sex );
  settings.setValue( attributeNames[9], row->birthDate.c_str() );
}


void StManagerStudyListComp::loadRecentRow ( DcmProvider::TableRow &row, const QSettings &settings )
{
  row.patientID = settings.value( attributeNames[0] ).toString().toStdString();
  row.name      = settings.value( attributeNames[1] ).toString().toStdString();
  row.modality  = settings.value( attributeNames[3] ).toString().toStdString();
  row.date      = settings.value( attributeNames[5] ).toString().toStdString();
  row.studyID   = settings.value( attributeNames[7] ).toString().toStdString();
  row.sex       = settings.value( attributeNames[8] ).toBool();
  row.birthDate = settings.value( attributeNames[9] ).toString().toStdString();
}


QTableWidget *StManagerStudyListComp::createStudyTable ()
{
  QTableWidget *table  = new QTableWidget;

  table->setSelectionBehavior( QAbstractItemView::SelectRows );
  table->setSelectionMode( QAbstractItemView::SingleSelection );
  table->setEditTriggers( QAbstractItemView::NoEditTriggers );

  QStringList labels;
  for ( int i = 0; i < ATTRIBUTE_NUMBER; i++ ) {
    labels << tr( attributeNames[i] );
  }
  
  table->setColumnCount( labels.size() + 1 );
  table->setHorizontalHeaderLabels( labels );
  table->setColumnHidden( ATTRIBUTE_NUMBER, true );

  connect( table, SIGNAL(itemSelectionChanged()), this, SLOT(setEnabledView()) );
  connect( table, SIGNAL(itemDoubleClicked(QTableWidgetItem *)), this, SLOT(view()) );

  return table;
}


QTreeView *StManagerStudyListComp::createDirectoryTreeView ()
{
  QTreeView *directoryTree = new QTreeView;

  QDirModel *model = new QDirModel();
  model->setFilter( QDir::Dirs | QDir::NoDotAndDotDot | QDir::Drives );

  directoryTree->setModel( model );
  directoryTree->setColumnWidth( 0, 150 );
  // hide size and type in filesystem tree view
  directoryTree->setColumnHidden( 1, true );
  directoryTree->setColumnHidden( 2, true );

  return directoryTree;
}


QPushButton *StManagerStudyListComp::createButton ( const QString &text, const char *member )
{
  QPushButton *button = new QPushButton( text );
  connect( button, SIGNAL(clicked()), this, member );

  return button;
}


QToolButton *StManagerStudyListComp::createToolButton ( const QIcon &icon, const char *member )
{
  QToolButton *toolButton = new QToolButton();
  toolButton->setCheckable( true );
  toolButton->setAutoExclusive( true );

  toolButton->setIconSize( QSize( 27, 27 ) );
  toolButton->setIcon( icon );

  connect( toolButton, SIGNAL(clicked()), this, member );

  return toolButton;
}

