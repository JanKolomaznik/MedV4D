#include "GUI/m4dGUIMainWindow.h"

#include <QtGui>

#include "Log.h"
#include "Debug.h"

// DICOM namespace:
using namespace M4D::Dicom;
using namespace M4D::Viewer;


/// Number of abstract viewer's actions (toolBar buttons) - first is for all unplugged slots (it's not in toolBar)
#define VIEWER_ACTIONS_NUMBER       13

const char *m4dGUIMainWindow::actionIconNames[] = { "empty.png", "window-level.png", "empty.png", "zoom.png", 
                                                    "stack.png", "info.png", "point.png", "shape.png", "clear-point.png",
                                                    "clear-shape.png", "clear-all.png", "flip-hor.png", "flip-vert.png" };

const char *m4dGUIMainWindow::actionTexts[] = { "No Action", "Window/Level (Right Mouse)", "Pan (Left Mouse)", 
                                                "Zoom (Right Mouse)", "Stack (Right Mouse)", "Toggle Overlay",
                                                "New Point (Left Mouse)", "New Shape (Left Mouse)", "Clear Point",
                                                "Clear Shape", "Clear All Points/Shapes", "Flip Horizontal", "Flip Vertical"};

// information also used for weather the connection is direct to the viewer
const bool  m4dGUIMainWindow::actionCheckables[] = { false, true, true, true, true, false, true, true, false, 
                                                     false, false, false, false };

const bool  m4dGUIMainWindow::actionRightButtons[] = { false, true, false, true, true, false, false, false, false,
                                                       false, false, false, false };

/* ?? */
const char *m4dGUIMainWindow::actionShortCuts[] = { "", "Ctrl+W", "Ctrl+P", "Ctrl+Z", "Ctrl+A", "Ctrl+T",
                                                    "Ctrl+I", "Ctrl+H", "Ctrl+N", "Ctrl+E", "Ctrl+A",
                                                    "Ctrl+R", "Ctrl+V" };

const char *m4dGUIMainWindow::actionStatusTips[] = { "Action for all unplugget available slots", 
                                                     "Adjust the brightness and/or contrast of the image", 
                                                     "Reposition the images in the window",
                                                     "Increase or decrease the image's field of view",
                                                     "Scroll through images within a series",
                                                     "Hide or display the study information",
                                                     "Create a new point",
                                                     "Start a new shape (end the previous one)",
                                                     "Clear last created point",
                                                     "Clear last created shape",
                                                     "Clear all selections (points, shapes)",
                                                     "Flip the selected image from left to right about the vertical axis",
                                                     "Flip the selected image from top to bottom about the horizontal axis" };

const char *m4dGUIMainWindow::actionSlots[] = { SLOT(empty()), SLOT(viewerWindowLevel()), SLOT(viewerPan()),
                                                SLOT(viewerZoom()), SLOT(viewerStack()), SLOT(slotTogglePrintData()),
                                                SLOT(viewerNewPoint()), SLOT(viewerNewShape()), SLOT(slotDeletePoint()),
                                                SLOT(slotDeleteShape()), SLOT(slotDeleteAll()), SLOT(slotToggleFlipHorizontal()),
                                                SLOT(slotToggleFlipVertical()) };

const int   m4dGUIMainWindow::slotsToActions[] = { 0, 0, 4, 0, 0, 12, 11, 0, 0, 0, 0, 0, 0, 5, 3, 2,
                                                   1, 6, 7, 8, 9, 10, 0, 0, 0 };

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

void m4dGUIMainWindow::size ()
{
  if ( smallAct->isChecked() ) {
    setIconSize( QSize( 24, 24 ) );
  }
  else if ( mediumAct->isChecked() ) {
    setIconSize( QSize( 36, 36 ) );
  }
  else {
    setIconSize( QSize( 64, 64 ) );
  }
}


void m4dGUIMainWindow::status ()
{
  if ( !statusBar()->isHidden() ) {
    statusBar()->hide();
  }
  else {
    statusBar()->show();
  }
}


void m4dGUIMainWindow::viewerWindowLevel ()
{
  emit toolChanged( m4dGUIAbstractViewerWidget::adjust_bc,
                    m4dGUIAbstractViewerWidget::right );
}


void m4dGUIMainWindow::viewerPan ()
{
  emit toolChanged( m4dGUIAbstractViewerWidget::moveI,
                    m4dGUIAbstractViewerWidget::left );
}


void m4dGUIMainWindow::viewerZoom ()
{
  emit toolChanged( m4dGUIAbstractViewerWidget::zoomI,
                    m4dGUIAbstractViewerWidget::right );
}


void m4dGUIMainWindow::viewerStack ()
{
  emit toolChanged( m4dGUIAbstractViewerWidget::switch_slice,
                    m4dGUIAbstractViewerWidget::right );
}


void m4dGUIMainWindow::viewerNewPoint ()
{
  emit toolChanged( m4dGUIAbstractViewerWidget::new_point,
                    m4dGUIAbstractViewerWidget::left );
}


void m4dGUIMainWindow::viewerNewShape ()
{
  emit toolChanged( m4dGUIAbstractViewerWidget::new_shape,
                    m4dGUIAbstractViewerWidget::left );
}


void m4dGUIMainWindow::features ()
{
  disconnect( this, 
              SIGNAL(toolChanged( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )), 
              mainViewerDesktop->getPrevSelectedViewer(),
              SLOT(slotSetButtonHandler( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )) );

  connect( this, 
           SIGNAL(toolChanged( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )), 
           mainViewerDesktop->getSelectedViewer(),
           SLOT(slotSetButtonHandler( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )) );

  for ( unsigned i = 0; i < VIEWER_ACTIONS_NUMBER; i++ ) 
  {
    if ( !actionCheckables[i] )
    {
      disconnect( viewerActs[i], SIGNAL(triggered()), mainViewerDesktop->getPrevSelectedViewer(), actionSlots[i] );
      connect( viewerActs[i], SIGNAL(triggered()), mainViewerDesktop->getSelectedViewer(), actionSlots[i] );
    }
  }

  m4dGUIAbstractViewerWidget *viewer = mainViewerDesktop->getSelectedViewer();
  m4dGUIAbstractViewerWidget::AvailableSlots availableFeatures = viewer->getAvailableSlots();

  for ( m4dGUIAbstractViewerWidget::AvailableSlots::iterator it = availableFeatures.begin(); 
        it != availableFeatures.end(); 
        it++ ) {
    viewerActs[slotsToActions[*it]]->setEnabled( true ); 
  }
}


void m4dGUIMainWindow::layout ()
{
  screenLayoutDialog->show();
}


void m4dGUIMainWindow::swap ()
{
  // mainViewerDesktop->getSelectedViewer(); 
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

  QActionGroup *toolBarSizeGroup  = new QActionGroup( this );

  smallAct = new QAction( tr( "Small" ), this );
  smallAct->setStatusTip( tr( "Set ToolBar size to small" ) );
  smallAct->setCheckable( true );
  connect( smallAct, SIGNAL(triggered()), this, SLOT(size()) );
  smallAct->setChecked( true );
  toolBarSizeGroup->addAction( smallAct );

  mediumAct = new QAction( tr( "Medium" ), this );
  mediumAct->setStatusTip( tr( "Set ToolBar size to medium" ) );
  mediumAct->setCheckable( true );
  connect( mediumAct, SIGNAL(triggered()), this, SLOT(size()) );
  toolBarSizeGroup->addAction( mediumAct );

  largeAct = new QAction( tr( "Large" ), this );
  largeAct->setStatusTip( tr( "Set ToolBar size to large" ) );
  largeAct->setCheckable( true );
  connect( largeAct, SIGNAL(triggered()), this, SLOT(size()) );
  toolBarSizeGroup->addAction( largeAct );

  statusAct = new QAction( tr( "Status Bar" ), this );
  statusAct->setStatusTip( tr( "Hide or show the Status Bar"  ) );
  statusAct->setCheckable( true );
  connect( statusAct, SIGNAL(triggered()), this, SLOT(status()) );
  statusAct->setChecked( true );

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
    
    viewerActs[i]->setEnabled( false );

    if ( actionCheckables[i] ) {
      connect( viewerActs[i], SIGNAL(triggered()), this, actionSlots[i] );
    }
    
    if ( actionCheckables[i] ) {
      actionRightButtons[i] ? rightButtonGroup->addAction( viewerActs[i] ) :
                              leftButtonGroup->addAction( viewerActs[i] );
    }
  }
  // update availability of features (according to selected viewer - first one is init.)
  features();

  layoutAct = new QAction( QIcon( ":/icons/layout.png" ), tr( "S&creen Layout" ), this );
  layoutAct->setShortcut( tr( "Ctrl+C" ) );
  layoutAct->setStatusTip( tr( "Redisplay series and images in various layouts" ) );
  connect( layoutAct, SIGNAL(triggered()), this, SLOT(layout()) );

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

  // ToolBars Menu
  toolBarsMenu = new QMenu( tr( "T&oolBars" ), this );
  toolBarsMenu->addAction( smallAct );
  toolBarsMenu->addAction( mediumAct );
  toolBarsMenu->addAction( largeAct );
  toolBarsMenu->addSeparator();
  toolBarsMenu->addAction( statusAct );

  menuBar()->addMenu( toolBarsMenu );

  // Tools Menu
  toolsMenu = new QMenu( tr( "&Tools" ), this );
  for ( unsigned i = 1; i < VIEWER_ACTIONS_NUMBER; i++ ) {
    toolsMenu->addAction( viewerActs[i] );
  }

  menuBar()->addMenu( toolsMenu );

  // View Menu
  // will be needed
  // viewMenu = new QMenu( tr( "&View" ), this );

  // menuBar()->addMenu( viewMenu );
}


void m4dGUIMainWindow::createToolBars ()
{
  searchToolBar = addToolBar( tr( "Search" ) );
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
