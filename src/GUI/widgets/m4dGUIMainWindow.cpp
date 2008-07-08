#include "GUI/m4dGUIMainWindow.h"

#include <QtGui>

#include "Log.h"
#include "Debug.h"

// DICOM namespace:
using namespace M4D::Dicom;


m4dGUIMainWindow::m4dGUIMainWindow ( const char *title, const QIcon &icon )
{
  Q_INIT_RESOURCE( m4dGUIMainWindow );

  QWidget *centralWidget = new QWidget;
  setCentralWidget( centralWidget );

  createVtkRenderWindowWidget();
  // just for pilot implementation
  vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->sphereToRenderWindow() );
  // vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->imageDataToRenderWindow() );

  createActions();
  createMenus();
  createToolBars();
  createStatusBar();
  createDockWindows();

  QVBoxLayout *mainLayout = new QVBoxLayout;
  mainLayout->addWidget( vtkRenderWindowWidget );
  centralWidget->setLayout( mainLayout );

  setWindowTitle( tr( title ) ); 
  setWindowIcon( icon );
  showMaximized();

  // everything is initialized, build the manager - it needs some of the previous steps.
  createStudyManagerDialog();
  // and the other dialogs
  createScreenLayoutDialog();
}


void m4dGUIMainWindow::search ()
{
  DcmProvider::DicomObjSet *dicomObjSet = new DcmProvider::DicomObjSet();	
  studyManagerWidget->getStudyListComponent()->setDicomObjectSet( dicomObjSet );

  // Study Manager Dialog - this will fill the dicomObjSet
  if ( studyManagerDialog->exec() ) {
    view( dicomObjSet );
  }
}


void m4dGUIMainWindow::open ()
{
  QString path( QFileDialog::getOpenFileName( this, tr( "Open File" ), 
                                              QDir::currentPath(), "*.dcm" ) );
  
  if ( !path.isNull() ) 
  {
    QFileInfo pathInfo( path );
    QString dirName( pathInfo.absolutePath() );

    vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->dicomToRenderWindow( dirName.toAscii() ) );
  } 
}


void m4dGUIMainWindow::layout ()
{
  screenLayoutDialog->show();
}


void m4dGUIMainWindow::createVtkRenderWindowWidget ()
{
  vtkRenderWindowWidget = new m4dGUIVtkRenderWindowWidget;
}


void m4dGUIMainWindow::createStudyManagerDialog ()
{
  // new dialog for Study Manager Widget - without What's This button in the title bar
  studyManagerDialog = new QDialog( this, Qt::WindowTitleHint | Qt::WindowSystemMenuHint );
  studyManagerDialog->setWindowTitle( tr( "Study Manager" ) );
  studyManagerDialog->setWindowIcon( QIcon( ":/icons/search.png" ) );

  studyManagerWidget = new m4dGUIStudyManagerWidget( vtkRenderWindowWidget, studyManagerDialog );

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( studyManagerWidget );
  
  studyManagerDialog->setLayout( dialogLayout );
}


void m4dGUIMainWindow::createScreenLayoutDialog ()
{
  // new dialog for Screen Layout Widget - without What's This button in the title bar and resize
  screenLayoutDialog = new QDialog( this, Qt::WindowTitleHint | Qt::WindowSystemMenuHint );

  screenLayoutWidget = new m4dGUIScreenLayoutWidget( vtkRenderWindowWidget,
                                                                               screenLayoutDialog );

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( screenLayoutWidget );

  screenLayoutDialog->setWindowTitle( tr( "Screen Layout" ) );
  screenLayoutDialog->setWindowIcon( QIcon( ":/icons/layout.png" ) );
  screenLayoutDialog->setLayout( dialogLayout );
}


void m4dGUIMainWindow::createActions ()
{
  searchAct = new QAction( QIcon( ":/icons/search.png" ), tr( "&Search" ), this );
  searchAct->setShortcut( tr( "Ctrl+S" ) );
  searchAct->setStatusTip( tr( "Search for patient studies available for viewing" ) );
  connect( searchAct, SIGNAL(triggered()), this, SLOT(search()) );

  openAct = new QAction( QIcon( ":/icons/open.png" ), tr( "&Open..." ), this );
  openAct->setShortcut( tr( "Ctrl+O" ) );
  openAct->setStatusTip( tr( "Open an existing document from disk or network file system" ) );
  connect( openAct, SIGNAL(triggered()), this, SLOT(open()) );

  saveAct = new QAction( QIcon( ":/icons/save.png" ), tr( "S&ave" ), this );
  saveAct->setShortcut( tr( "Ctrl+A" ) );
  saveAct->setStatusTip( tr( "Save the document to disk" ) );
  connect( saveAct, SIGNAL(triggered()), this, SLOT(save()) );
  saveAct->setEnabled( false );

  exitAct = new QAction( QIcon( ":/icons/exit.png" ), tr( "E&xit" ), this );
  exitAct->setShortcut( tr( "Ctrl+Q" ) );
  connect( exitAct, SIGNAL(triggered()), this, SLOT(close()) );

  layoutAct = new QAction( QIcon( ":/icons/layout.png" ), tr( "S&creen Layout" ), this );
  layoutAct->setShortcut( tr( "Ctrl+C" ) );
  layoutAct->setStatusTip( tr( "Redisplay series and images in various layouts" ) );
  connect( layoutAct, SIGNAL(triggered()), this, SLOT(layout()) );

  overlayAct = new QAction( QIcon( ":/icons/info.png" ), tr( "&Toggle Overlay" ), this );
  overlayAct->setShortcut( tr( "Ctrl+T" ) );
  overlayAct->setStatusTip( tr( "Hide or displays the study information" ) );
  connect( overlayAct, SIGNAL(triggered()), this, SLOT(overlay()) );
  overlayAct->setEnabled( false );
}


void m4dGUIMainWindow::createMenus ()
{
  // File Menu
  fileMenu = new QMenu( tr( "&File" ), this );
  fileMenu->addAction( searchAct );
  fileMenu->addAction( openAct );
  fileMenu->addAction( saveAct );
  fileMenu->addSeparator();
  fileMenu->addAction( exitAct );

  menuBar()->addMenu( fileMenu );

  // View Menu
  viewMenu = new QMenu( tr( "&View" ), this );

  menuBar()->addMenu( viewMenu );
}


void m4dGUIMainWindow::createToolBars ()
{
  searchToolBar = addToolBar( tr( "Search" ) );
  searchToolBar->addAction( searchAct );

  fileToolBar = addToolBar( tr( "File" ) );
  fileToolBar->addAction( openAct );
  fileToolBar->addAction( saveAct );

  viewToolBar = addToolBar( tr( "View" ) );
  viewToolBar->addAction( layoutAct );
  viewToolBar->addAction( overlayAct );
}


void m4dGUIMainWindow::createStatusBar()
{
  statusBar()->showMessage( tr( "Ready" ) );
}


void m4dGUIMainWindow::createDockWindows ()
{
  QDockWidget *dock = new QDockWidget( tr( "DockWidget One" ), this );
  dock->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
  
  QListWidget *listOne = new QListWidget( dock );
  dock->setWidget( listOne );
 
  addDockWidget( Qt::RightDockWidgetArea, dock );
  viewMenu->addAction( dock->toggleViewAction() );

  dock = new QDockWidget( tr( "DockWidget Two" ), this );
  
  QListWidget *listTwo = new QListWidget( dock );
  dock->setWidget( listTwo );

  addDockWidget( Qt::RightDockWidgetArea, dock );
  viewMenu->addAction( dock->toggleViewAction() );

  dock = new QDockWidget( tr( "DockWidget Three" ), this );
  
  QListWidget *listThree = new QListWidget( dock );
  dock->setWidget( listThree );

  addDockWidget( Qt::RightDockWidgetArea, dock );
  viewMenu->addAction( dock->toggleViewAction() );
}


void m4dGUIMainWindow::view ( DcmProvider::DicomObjSet *dicomObjSet )
{
  vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->imageDataToRenderWindow( DcmProvider::DicomObjSetPtr( dicomObjSet ) ) );
}
