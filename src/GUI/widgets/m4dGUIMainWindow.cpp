#include "GUI/m4dGUIMainWindow.h"

#include <QtGui>

#include "Log.h"
#include "Debug.h"

// DICOM namespace:
using namespace M4D::Dicom;
using namespace M4D::Viewer;


/// Number of abstract viewer's actions (toolBar buttons) - first is for all unplugged slots (it's not in toolBar)
#define VIEWER_ACTIONS_NUMBER       4

const char *m4dGUIMainWindow::actionIconNames[] = { "empty.png", "window-level.png", "zoom.png", "info.png"  };

const char *m4dGUIMainWindow::actionTexts[] = { "No Action", "&Window/Level (Right Mouse)", "&Zoom (Right Mouse)", 
                                                "&Toggle Overlay"  };

const bool  m4dGUIMainWindow::actionCheckables[] = { false, true, true, false };

const char *m4dGUIMainWindow::actionShortCuts[] = { "", "Ctrl+W", "Ctrl+Z", "Ctrl+T"  };

const char *m4dGUIMainWindow::actionStatusTips[] = { "Action for all unplugget available slots", 
                                                     "Adjust the brightness and/or contrast of the image", 
                                                     "Increase or decrease the image's field of view",
                                                     "Hide or displays the study information"  };

const char *m4dGUIMainWindow::actionSlots[] = { SLOT(empty()), SLOT(slotWindowLevel()), SLOT(slotZoom()), 
                                                SLOT(slotOverlay()) };

const bool  m4dGUIMainWindow::actionRightButtons[] = { false, true, true, false };

const int   m4dGUIMainWindow::slotsToActions[] = { 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3 };

m4dGUIMainWindow::m4dGUIMainWindow ( const char *title, const QIcon &icon )
{
  Q_INIT_RESOURCE( m4dGUIMainWindow );

  QWidget *centralWidget = new QWidget;
  setCentralWidget( centralWidget );

  createMainViewerDesktop();

  createActions();
  createMenus();
  createToolBars();
  createStatusBar();
  // will be needed
  // createDockWindows();

  QVBoxLayout *mainLayout = new QVBoxLayout;
  mainLayout->setContentsMargins( 0, 0, 0, 0 );
  mainLayout->addWidget( mainViewerDesktop );
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

    mainViewerDesktop->getVtkRenderWindowWidget()->addRenderer( mainViewerDesktop->getVtkRenderWindowWidget()->dicomToRenderWindow( dirName.toAscii() ) );
  } 
}


void m4dGUIMainWindow::layout ()
{
  screenLayoutDialog->show();
}


void m4dGUIMainWindow::features ()
{
  m4dAbstractViewerWidget *viewer = mainViewerDesktop->getSelectedViewer();
  m4dAbstractViewerWidget::AvailableSlots availableFeatures = viewer->getAvailableSlots();

  for ( m4dAbstractViewerWidget::AvailableSlots::iterator it = availableFeatures.begin(); 
        it != availableFeatures.end(); 
        it++ ) {
    viewerActs[slotsToActions[*it]]->setEnabled( true ); 
  }
}


void m4dGUIMainWindow::createMainViewerDesktop ()
{
  mainViewerDesktop = new m4dGUIMainViewerDesktopWidget;
  connect( mainViewerDesktop, SIGNAL(propagateFeatures()), this, SLOT(features()) );
}


void m4dGUIMainWindow::createStudyManagerDialog ()
{
  // new dialog for Study Manager Widget - without What's This button in the title bar
  studyManagerDialog = new QDialog( this, Qt::WindowTitleHint | Qt::WindowSystemMenuHint );
  studyManagerDialog->setWindowTitle( tr( "Study Manager" ) );
  studyManagerDialog->setWindowIcon( QIcon( ":/icons/search.png" ) );

  studyManagerWidget = new m4dGUIStudyManagerWidget( studyManagerDialog );

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( studyManagerWidget );
  
  studyManagerDialog->setLayout( dialogLayout );
}


void m4dGUIMainWindow::createScreenLayoutDialog ()
{
  // new dialog for Screen Layout Widget - without What's This button in the title bar and resize
  screenLayoutDialog = new QDialog( this, Qt::WindowTitleHint | Qt::WindowSystemMenuHint | Qt::MSWindowsFixedSizeDialogHint );
  screenLayoutDialog->setWindowTitle( tr( "Screen Layout" ) );
  screenLayoutDialog->setWindowIcon( QIcon( ":/icons/layout.png" ) );

  screenLayoutWidget = new m4dGUIScreenLayoutWidget( mainViewerDesktop, screenLayoutDialog );

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( screenLayoutWidget );

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

  QActionGroup *leftButtonGroup  = new QActionGroup( this );
  QActionGroup *rightButtonGroup = new QActionGroup( this );

  viewerActs = new QAction *[VIEWER_ACTIONS_NUMBER];
  for ( unsigned i = 0; i < VIEWER_ACTIONS_NUMBER; i++ ) 
  {
    QString fileName = QString( ":/icons/" ).append( actionIconNames[i] );
    viewerActs[i] = new QAction( QIcon( fileName ), tr( actionTexts[i] ), this );
    viewerActs[i]->setCheckable( actionCheckables[i] );
    viewerActs[i]->setShortcut( tr( actionShortCuts[i] ) );
    viewerActs[i]->setStatusTip( tr( actionStatusTips[i] ) );
    
    connect( viewerActs[i], SIGNAL(triggered()), mainViewerDesktop->getSelectedViewer(), actionSlots[i] );
    
    viewerActs[i]->setEnabled( false );
    
    actionRightButtons[i] && actionCheckables[i] ? rightButtonGroup->addAction( viewerActs[i] ) :
                                                   leftButtonGroup->addAction( viewerActs[i] );
  }
  // update availability of features (according to selected viewer - first one is init.)
  features();

  swapAct = new QAction( QIcon( ":/icons/swap.png" ), tr( "Swa&p Viewers" ), this );
  swapAct->setShortcut( tr( "Ctrl+P" ) );
  swapAct->setStatusTip( tr( "Swap selected viewer" ) );
  connect( layoutAct, SIGNAL(triggered()), this, SLOT(swap()) );
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
  // will be needed
  // viewMenu = new QMenu( tr( "&View" ), this );

  // menuBar()->addMenu( viewMenu );
}


void m4dGUIMainWindow::createToolBars ()
{
  searchToolBar = addToolBar( tr( "Search" ) );
  // searchToolBar->setIconSize( QSize( 34, 34 ) );
  searchToolBar->addAction( searchAct );

  fileToolBar = addToolBar( tr( "File" ) );
  fileToolBar->addAction( openAct );
  fileToolBar->addAction( saveAct );

  layoutToolBar = addToolBar( tr( "Layout" ) );
  layoutToolBar->addAction( layoutAct );

  viewerToolBar = addToolBar( tr( "Viewer" ) );
  for ( unsigned i = 1; i < VIEWER_ACTIONS_NUMBER; i++ ) {
    viewerToolBar->addAction( viewerActs[i] );
  }

  swapToolBar = addToolBar( tr( "Swap" ) );
  swapToolBar->addAction( swapAct );
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
  mainViewerDesktop->getVtkRenderWindowWidget()->addRenderer( mainViewerDesktop->getVtkRenderWindowWidget()->imageDataToRenderWindow( DcmProvider::DicomObjSetPtr( dicomObjSet ) ) );
}
