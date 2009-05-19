/**
 *  @ingroup gui
 *  @file m4dGUIMainWindow.cpp
 *  @brief some brief
 */
#include "GUI/widgets/m4dGUIMainWindow.h"

#include <QtGui>

#include "Imaging/Imaging.h"
#include "GUI/widgets/utils/LoadingThreads.h"

using namespace M4D::Dicom;
using namespace M4D::Viewer;
using namespace M4D::Imaging;
using namespace M4D::IO;

using namespace std;


// Q_INIT_RESOURCE macro cannot be used in a namespace
inline void initMainWindowResource () { Q_INIT_RESOURCE( m4dGUIMainWindow ); }

namespace M4D {
namespace GUI {

/// Number of abstract viewer's actions (toolBar buttons) - first is for all unplugged slots (it's not in toolBar)
#define VIEWER_ACTIONS_NUMBER       16

/// Names of the action icons
const char *m4dGUIMainWindow::actionIconNames[] = { "empty.png", "window-level.png", "move.png", "zoom.png",
                  "stack.png", "info.png", "probe.png", "point.png", "shape.png", "clear-point.png",
                  "clear-shape.png", "clear-all.png", "flip-hor.png", "flip-vert.png", "slice-orient.png",
                  "rotate-volume.png" };

/// Texts of the actions
const char *m4dGUIMainWindow::actionTexts[] = { "No Action", "Window/Level (Right Mouse)", "Pan (Left Mouse)",
                  "Zoom (Right Mouse)", "Stack (Right Mouse)", "Toggle Overlay", "Probe Tool (Left Mouse)", 
                  "New Point (Left Mouse)", "New Shape (Left Mouse)", "Clear Point", "Clear Shape", 
                  "Clear All Points/Shapes", "Flip Horizontal", "Flip Vertical", "Slice Orientation",
                  "Rotate Volume (Left Mouse)" };

/// Types of the actions - information also used for weather the connection is direct to the viewer
const m4dGUIMainWindow::ToolType m4dGUIMainWindow::actionToolTypes[] = { CHECKABLE_TOOL, CHECKABLE_TOOL, CHECKABLE_TOOL,
                  CHECKABLE_TOOL, CHECKABLE_TOOL, TOGGLE_TOOL, CHECKABLE_TOOL, CHECKABLE_TOOL, CHECKABLE_TOOL, 
                  TOGGLE_TOOL, TOGGLE_TOOL, TOGGLE_TOOL, TOGGLE_TOOL, TOGGLE_TOOL, TOGGLE_TOOL, CHECKABLE_TOOL };

/// Types of the mouse button for the actions (checkable tools)
const m4dGUIMainWindow::ButtonType m4dGUIMainWindow::actionButtonTypes[] = { LEFT_BUTTON, RIGHT_BUTTON, LEFT_BUTTON, 
                  RIGHT_BUTTON, RIGHT_BUTTON, LEFT_BUTTON, LEFT_BUTTON, LEFT_BUTTON, LEFT_BUTTON, LEFT_BUTTON, 
                  LEFT_BUTTON, LEFT_BUTTON, LEFT_BUTTON, LEFT_BUTTON, LEFT_BUTTON, LEFT_BUTTON };

/// Shortcuts of the actions
const char *m4dGUIMainWindow::actionShortCuts[] = { "", "Ctrl+W", "Ctrl+P", "Ctrl+Z", "Ctrl+K", "Ctrl+T",
                  "Ctrl+B", "Ctrl+I", "Ctrl+H", "Ctrl+N", "Ctrl+E", "Ctrl+A", "Ctrl+R", "Ctrl+V", "Ctrl+G",
                  "Ctrl+U" };

/// StatusTips of the actions
const char *m4dGUIMainWindow::actionStatusTips[] = { "Action for all unplugged available slots", 
                                                     "Adjust the brightness and/or contrast of the image", 
                                                     "Reposition the images in the window",
                                                     "Increase or decrease the image's field of view",
                                                     "Scroll through images within a series",
                                                     "Hide or display the study information",
                                                     "Give a pixel value for a given point",
                                                     "Create a new point",
                                                     "Start a new shape (end the previous one)",
                                                     "Clear last created point",
                                                     "Clear last created shape",
                                                     "Clear all selections (points, shapes)",
                                                     "Flip the selected image from left to right about the vertical axis",
                                                     "Flip the selected image from top to bottom about the horizontal axis",
                                                     "Switch the viewing axis orientation (x~y, y~z, z~x)",
                                                     "Rotates the volume about the three axes" };

/// Slots for the actions
const char *m4dGUIMainWindow::actionSlots[] = { SLOT(empty()), SLOT(viewerWindowLevel()), SLOT(viewerPan()),
                  SLOT(viewerZoom()), SLOT(viewerStack()), SLOT(slotTogglePrintData()), SLOT(viewerProbe()), 
                  SLOT(viewerNewPoint()), SLOT(viewerNewShape()), SLOT(slotDeletePoint()), SLOT(slotDeleteShape()),
                  SLOT(slotDeleteAll()), SLOT(slotToggleFlipHorizontal()), SLOT(slotToggleFlipVertical()), 
                  SLOT(slotToggleSliceOrientation()), SLOT(viewerRotate()) };

/// Mapping available slots of viewers to actions
const int m4dGUIMainWindow::slotsToActions[] = { ACTION_EMPTY, ACTION_EMPTY, ACTION_STACK, ACTION_EMPTY, ACTION_EMPTY,
                  ACTION_FLIP_VERTICAL, ACTION_FLIP_HORIZONTAL, ACTION_EMPTY, ACTION_EMPTY, ACTION_EMPTY, ACTION_EMPTY, 
                  ACTION_EMPTY, ACTION_EMPTY, ACTION_OVERLAY, ACTION_EMPTY, ACTION_ZOOM, ACTION_PAN, ACTION_WINDOW_LEVEL, 
                  ACTION_NEW_POINT, ACTION_NEW_SHAPE, ACTION_CLEAR_POINT, ACTION_CLEAR_SHAPE, ACTION_CLEAR_ALL, 
                  ACTION_ROTATE_3D, ACTION_ROTATE_3D, ACTION_ROTATE_3D, ACTION_SLICE_ORIENTATION, ACTION_PROBE };

m4dGUIMainWindow::m4dGUIMainWindow ( const char *appName, const char *orgName, const QIcon &icon )
{
  QCoreApplication::setApplicationName( appName );
  QCoreApplication::setOrganizationName( orgName );

  initMainWindowResource();

  mainDesktopStackedWidget = new QStackedWidget();
  // reserve the first slot to simple widget
  mainDesktopStackedWidget->addWidget( new QWidget );
  setCentralWidget( mainDesktopStackedWidget );

  createDefaultViewerDesktop();
  addViewerDesktop( currentViewerDesktop ); 
  mainDesktopStackedWidget->setCurrentIndex( 1 );

  createActions();
  createMenus();
  createToolBars();
  createStatusBar();

  setWindowTitle( tr( appName ) ); 
  setWindowIcon( icon );
  showMaximized();

  // everything is initialized, build the manager - it needs some of the previous steps.
  createStudyManagerDialog();
  // and the other dialogs
  createScreenLayoutDialog();
  createToolBarCustomizerDialog();
  createProgressBarDialog();

  // update availability of features (according to selected viewer - first one is init.)
  features( currentViewerDesktop->getPrevSelectedViewerWidget() );
}


void m4dGUIMainWindow::addDesktopWidget ( QWidget *widget )
{
  // always add it to the reserved first place
  mainDesktopStackedWidget->removeWidget( mainDesktopStackedWidget->widget( 0 ) );
  mainDesktopStackedWidget->insertWidget( 0, widget );
}


int m4dGUIMainWindow::addViewerDesktop ( m4dGUIMainViewerDesktopWidget *viewerDesktop )
{
  connect( viewerDesktop, SIGNAL(propagateFeatures( M4D::Viewer::m4dGUIAbstractViewerWidget * )), 
           this, SLOT(features( M4D::Viewer::m4dGUIAbstractViewerWidget * )) );

  return mainDesktopStackedWidget->addWidget( viewerDesktop );
}


void m4dGUIMainWindow::switchToDesktopWidget ()
{
  mainDesktopStackedWidget->setCurrentIndex( 0 );

  sourcesToolBar->hide();

  for ( unsigned i = 0; i < VIEWER_ACTIONS_NUMBER; i++ ) {
    viewerActs[i]->setEnabled( false ); 
  }
  replaceAct->setEnabled( false );
  layoutAct->setEnabled( false );
}


void m4dGUIMainWindow::switchToViewerDesktop ( int index, bool enableOtherTools )
{
  mainDesktopStackedWidget->setCurrentIndex( index );

  m4dGUIMainViewerDesktopWidget *prevViewerDesktop = currentViewerDesktop;
  currentViewerDesktop = (m4dGUIMainViewerDesktopWidget *)mainDesktopStackedWidget->currentWidget();

  if ( sourcesComboBox->count() ) 
  {
    sourcesToolBar->show();

    sourcesComboBox->setCurrentIndex( currentViewerDesktop->getSelectedViewerSourceIdx() );
  }

  features( prevViewerDesktop->getSelectedViewerWidget() );
  replaceAct->setEnabled( enableOtherTools );
  layoutAct->setEnabled( enableOtherTools );
}


void m4dGUIMainWindow::switchToDefaultViewerDesktop ()
{
  switchToViewerDesktop( 1 );
}


void m4dGUIMainWindow::addSource ( ConnectionInterface *conn, 
                                   const char *pipelineDescription, const char *connectionDescription,
                                   m4dGUIViewerEventHandlerInterface *viewerEventHandler )
{
  for ( unsigned i = 1; i < mainDesktopStackedWidget->count(); i++ ) 
  {
    m4dGUIMainViewerDesktopWidget *desktop = (m4dGUIMainViewerDesktopWidget *)mainDesktopStackedWidget->widget( i );
    desktop->addSource( conn, viewerEventHandler );
  }

  sourcesComboBox->addItem( QString( pipelineDescription ) + " - " + QString( connectionDescription ) );

  sourcesToolBar->show();
}


void m4dGUIMainWindow::addDockWindow ( const char *title, QWidget *widget, DockWindowType windowType )
{
  QDockWidget *dock = new QDockWidget( tr( title ), this );

  dock->setWidget( widget );

  dock->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
  if ( windowType == FLOATING_DOCK_WINDOW )
  {
    dock->setFloating( true );
    dock->resize( dock->sizeHint() );
    dock->move( x() + width() - dock->width() - 30, y() + 170 );
  }
  
  addDockWidget( Qt::RightDockWidgetArea, dock );
  viewMenu->addAction( dock->toggleViewAction() );

  menuBar()->addMenu( viewMenu );
}


void m4dGUIMainWindow::search ()
{
  actualStudy.dicomObjSet = new DicomObjSet();	
  studyManagerWidget->getStudyListComponent()->setDicomObjectSetPtr( actualStudy.dicomObjSet );
  
  actualStudy.leftOverlayInfo.clear();
  actualStudy.rightOverlayInfo.clear();

  // Study Manager Dialog - this will fill the dicomObjSet, overlay infos
  studyManagerDialog->show(); 
}


void m4dGUIMainWindow::open ()
{
  actualStudy.leftOverlayInfo.clear();
  actualStudy.rightOverlayInfo.clear();

  QString path( QFileDialog::getOpenFileName( this, tr( "Open" ), currentOpenPath, "*.dcm" ) );
  
  if ( !path.isNull() ) 
  {
    startProgress();

    currentOpenPath = path;
    QFileInfo pathInfo( currentOpenPath );

    actualStudy.dicomObjSet = new DicomObjSet();	

    Multithreading::Thread loadingThread( OpenLoadingThread ( pathInfo.absoluteFilePath().toStdString(), 
  		                                                        pathInfo.absolutePath().toStdString(),
  		                                                        actualStudy.dicomObjSet, this ) );
  }
}


void m4dGUIMainWindow::load ()
{
  QString path( QFileDialog::getOpenFileName( this, tr( "Load" ), "", "*.mv4d" ) );
  
  if ( !path.isNull() ) 
  {
    QFileInfo pathInfo( path );

    FInStream inStr( pathInfo.absoluteFilePath().toStdString() );		
    actualStudy.abstractDataSet = DataSetFactory::DeserializeDataset( inStr );

    process( actualStudy.abstractDataSet );
  } 
}


void m4dGUIMainWindow::save ()
{
  QString path( QFileDialog::getSaveFileName( this, tr( "Save" ), "", "*.mv4d" ) );
  
  if ( !path.isNull() ) 
  {
    QFileInfo pathInfo( path );

    FOutStream stream( pathInfo.absoluteFilePath().toStdString() );

    try {
      DataSetFactory::SerializeDataset( stream, currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].GetDataset() );
    }
    catch ( Port::EDisconnected &e ) {
      QMessageBox::critical( this, tr( "Exception" ), e.what() ); 
    }
  }
}


void m4dGUIMainWindow::customize ()
{
  toolBarCustomizerDialog->show();
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
  delegateAction( ACTION_WINDOW_LEVEL, m4dGUIAbstractViewerWidget::adjust_bc );
}


void m4dGUIMainWindow::viewerPan ()
{
  delegateAction( ACTION_PAN, m4dGUIAbstractViewerWidget::moveI );
}


void m4dGUIMainWindow::viewerZoom ()
{  
  delegateAction( ACTION_ZOOM, m4dGUIAbstractViewerWidget::zoomI );
}


void m4dGUIMainWindow::viewerStack ()
{
  delegateAction( ACTION_STACK, m4dGUIAbstractViewerWidget::switch_slice );
}


void m4dGUIMainWindow::viewerProbe ()
{
  delegateAction( ACTION_PROBE, m4dGUIAbstractViewerWidget::color_picker );
}


void m4dGUIMainWindow::viewerNewPoint ()
{
  delegateAction( ACTION_NEW_POINT, m4dGUIAbstractViewerWidget::new_point );
}


void m4dGUIMainWindow::viewerNewShape ()
{
  delegateAction( ACTION_NEW_SHAPE, m4dGUIAbstractViewerWidget::new_shape );
}


void m4dGUIMainWindow::viewerRotate ()
{
  delegateAction( ACTION_ROTATE_3D, m4dGUIAbstractViewerWidget::rotate_3D );
}


void m4dGUIMainWindow::features ( m4dGUIAbstractViewerWidget *prevViewer )
{
  disconnect( this, 
              SIGNAL(toolChanged( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )), 
              prevViewer,
              SLOT(slotSetButtonHandler( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )) );

  m4dGUIAbstractViewerWidget *actViewer = currentViewerDesktop->getSelectedViewerWidget();
  connect( this, 
           SIGNAL(toolChanged( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )), 
           actViewer,
           SLOT(slotSetButtonHandler( m4dGUIAbstractViewerWidget::ButtonHandler, m4dGUIAbstractViewerWidget::MouseButton )) );

  for ( unsigned i = 0; i < VIEWER_ACTIONS_NUMBER; i++ ) 
  {
    viewerActs[i]->setEnabled( false ); 

    if ( actionToolTypes[i] == TOGGLE_TOOL )
    {
      disconnect( viewerActs[i], SIGNAL(triggered()), prevViewer, actionSlots[i] );
      connect( viewerActs[i], SIGNAL(triggered()), actViewer, actionSlots[i] );
    }
  }
  // new point/shape behavior
  disconnect( prevViewer, SIGNAL(signalNewShape( unsigned, double, double, double )), 
              viewerActs[ACTION_NEW_POINT], SLOT(trigger()) );
  connect( actViewer, SIGNAL(signalNewShape( unsigned, double, double, double )), 
           viewerActs[ACTION_NEW_POINT], SLOT(trigger()) );

  disconnect( screenLayoutWidget, SIGNAL(imageLayout()), prevViewer, SLOT(slotSetOneSliceMode()) );
  disconnect( screenLayoutWidget, SIGNAL(imageLayout( const unsigned, const unsigned )), 
              prevViewer, SLOT(slotSetMoreSliceMode( unsigned, unsigned )) );
  connect( screenLayoutWidget, SIGNAL(imageLayout()), actViewer, SLOT(slotSetOneSliceMode()) );
  connect( screenLayoutWidget, SIGNAL(imageLayout( const unsigned, const unsigned )), 
           actViewer, SLOT(slotSetMoreSliceMode( unsigned, unsigned )) );

  m4dGUIAbstractViewerWidget::AvailableSlots availableFeatures = actViewer->getAvailableSlots();

  for ( m4dGUIAbstractViewerWidget::AvailableSlots::iterator it = availableFeatures.begin(); 
        it != availableFeatures.end(); 
        it++ ) {
    viewerActs[slotsToActions[*it]]->setEnabled( true ); 
  }

  // trigger previously checked tools for newly selected viewer
  viewerActs[currentViewerDesktop->getSelectedViewerLeftTool()]->trigger();
  viewerActs[currentViewerDesktop->getSelectedViewerRightTool()]->trigger();

  if ( dynamic_cast< M4D::Viewer::m4dGUIVtkViewerWidget* >( currentViewerDesktop->getSelectedViewerWidget() ) ) {
    replaceAct->setChecked( true );
  }
  else {
    replaceAct->setChecked( false );
  }

  sourcesComboBox->setCurrentIndex( currentViewerDesktop->getSelectedViewerSourceIdx() );
}


void m4dGUIMainWindow::layout ()
{
  screenLayoutDialog->show();
}


void m4dGUIMainWindow::replace ()
{
  m4dGUIAbstractViewerWidget *replacedViewer = currentViewerDesktop->getSelectedViewerWidget();
  
  if ( !replaceAct->isChecked() ) {
    currentViewerDesktop->replaceSelectedViewerWidget( &SliceViewerFactory(),
                                                       replacedViewer );
  }
  else {
    currentViewerDesktop->replaceSelectedViewerWidget( &VtkViewerFactory(),
                                                       replacedViewer );
  }

  features( currentViewerDesktop->getPrevSelectedViewerWidget() );
}


void m4dGUIMainWindow::startProgress ()
{
  progressBarDialog->show();
  progressBarWidget->start();
}


void m4dGUIMainWindow::stopProgress ()
{
  progressBarWidget->stop();
}


void m4dGUIMainWindow::loadingReady ()
{
  if ( !actualStudy.dicomObjSet->empty() ) 
  {
    currentViewerDesktop->getSelectedViewerWidget()->setLeftSideTextData( actualStudy.leftOverlayInfo );  
    currentViewerDesktop->getSelectedViewerWidget()->setRightSideTextData( actualStudy.rightOverlayInfo );  

    actualStudy.abstractDataSet = DcmProvider::CreateImageFromDICOM( DicomObjSetPtr( actualStudy.dicomObjSet ) );

    process( actualStudy.abstractDataSet );

    stopProgress();
  }
  else {
    loadingException ( tr( "Empty image - nothing to process" ) );
  } 
}


void m4dGUIMainWindow::loadingException ( const QString &description )
{
  delete actualStudy.dicomObjSet;

  stopProgress();

  QMessageBox::critical( this, tr( "Exception" ), description );
}


void m4dGUIMainWindow::createDefaultViewerDesktop ()
{
  currentViewerDesktop = new m4dGUIMainViewerDesktopWidget( 1, 2, new SliceViewerFactory() );
}


void m4dGUIMainWindow::createStudyManagerDialog ()
{
  // new dialog for Study Manager Widget - without What's This button in the title bar
  studyManagerDialog = new QDialog( this, Qt::WindowTitleHint | Qt::WindowSystemMenuHint );
  studyManagerDialog->setWindowTitle( tr( "Study Manager" ) );
  studyManagerDialog->setWindowIcon( QIcon( ":/icons/search.png" ) );
  studyManagerDialog->setWindowModality( Qt::WindowModal );

  studyManagerWidget = new m4dGUIStudyManagerWidget( studyManagerDialog );
  connect( studyManagerWidget->getStudyListComponent(), SIGNAL(ready()), studyManagerDialog, SLOT(accept()) );
  connect( studyManagerWidget->getStudyListComponent(), SIGNAL(cancel()), studyManagerDialog, SLOT(reject()) );
  connect( studyManagerWidget->getStudyListComponent(), SIGNAL(start()), studyManagerDialog, SLOT(hide()) );

  connect( studyManagerWidget->getStudyListComponent(), SIGNAL(ready()), this, SLOT(loadingReady()) );
  connect( studyManagerWidget->getStudyListComponent(), SIGNAL(exception( const QString & )), this, SLOT(loadingException( const QString & )) );
 
  connect( studyManagerWidget->getStudyListComponent(), SIGNAL(start()), this, SLOT(startProgress()) );

  studyManagerWidget->getStudyListComponent()->setOverlayInfoPtr( &actualStudy.leftOverlayInfo,
                                                                  &actualStudy.rightOverlayInfo );

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( studyManagerWidget );
  
  studyManagerDialog->setLayout( dialogLayout );
}


void m4dGUIMainWindow::createToolBarCustomizerDialog ()
{
  // new dialog for ToolBar Customizer Widget - without What's This button in the title bar
  toolBarCustomizerDialog = new QDialog( this, Qt::WindowTitleHint | Qt::WindowSystemMenuHint );
  toolBarCustomizerDialog->setWindowTitle( tr( "Viewer Tool Property" ) );

  toolBarCustomizerWidget = new m4dGUIToolBarCustomizerWidget( viewerActs, VIEWER_ACTIONS_NUMBER );
  connect( toolBarCustomizerWidget, SIGNAL(ready()), toolBarCustomizerDialog, SLOT(accept()) );
  connect( toolBarCustomizerWidget, SIGNAL(cancel()), toolBarCustomizerDialog, SLOT(reject()) );

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( toolBarCustomizerWidget );
  
  toolBarCustomizerDialog->setLayout( dialogLayout );

  toolBarCustomizerDialog->resize( 330, 450 );
}


void m4dGUIMainWindow::createScreenLayoutDialog ()
{
  // new dialog for Screen Layout Widget - without What's This button in the title bar and resize
  screenLayoutDialog = new QDialog( this, Qt::WindowTitleHint | Qt::WindowSystemMenuHint | Qt::MSWindowsFixedSizeDialogHint );
  screenLayoutDialog->setWindowTitle( tr( "Screen Layout" ) );
  screenLayoutDialog->setWindowIcon( QIcon( ":/icons/layout.png" ) );

  screenLayoutWidget = new m4dGUIScreenLayoutWidget;
  connect( screenLayoutWidget, SIGNAL(seriesLayout( const unsigned, const unsigned )), 
           currentViewerDesktop, SLOT(setDesktopLayout( const unsigned, const unsigned )) );
  connect( screenLayoutWidget, SIGNAL(ready()), screenLayoutDialog, SLOT(close()) );

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( screenLayoutWidget );

  screenLayoutDialog->setLayout( dialogLayout );
}


void m4dGUIMainWindow::createProgressBarDialog ()
{
  // new dialog for Progress Bar Widget - fixed, just with title
  progressBarDialog = new QDialog( this, Qt::Dialog | Qt::WindowTitleHint | Qt::MSWindowsFixedSizeDialogHint );
  progressBarDialog->setWindowTitle( tr( "Loading..." ) );
  progressBarDialog->setWindowModality( Qt::WindowModal );

  progressBarWidget = new m4dGUIProgressBarWidget( tr( "Loading Data Sets..." ) );
  connect( progressBarWidget, SIGNAL(ready()), progressBarDialog, SLOT(close()) );
  connect( progressBarDialog, SIGNAL(rejected()), progressBarWidget, SLOT(pause()) );
  
  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( progressBarWidget );
  
  progressBarDialog->setLayout( dialogLayout );
}


void m4dGUIMainWindow::createActions ()
{
  searchAct = new QAction( QIcon( ":/icons/search.png" ), tr( "&Search" ), this );
  searchAct->setShortcut( tr( "Ctrl+S" ) );
  searchAct->setStatusTip( tr( "Search for patient studies available for viewing" ) );
  connect( searchAct, SIGNAL(triggered()), this, SLOT(search()) );

  openAct = new QAction( QIcon( ":/icons/open.png" ), tr( "&Open..." ), this );
  openAct->setShortcut( tr( "Ctrl+O" ) );
  openAct->setStatusTip( tr( "Open an existing document (DICOM) from disk or network file system" ) );
  connect( openAct, SIGNAL(triggered()), this, SLOT(open()) );

  loadAct = new QAction( QIcon( ":/icons/load.png" ), tr( "Loa&d..." ), this );
  loadAct->setShortcut( tr( "Ctrl+D" ) );
  loadAct->setStatusTip( tr( "Load saved Data Set" ) );
  connect( loadAct, SIGNAL(triggered()), this, SLOT(load()) );

  saveAct = new QAction( QIcon( ":/icons/save.png" ), tr( "S&ave..." ), this );
  saveAct->setShortcut( tr( "Ctrl+A" ) );
  saveAct->setStatusTip( tr( "Save current Data Set" ) );
  connect( saveAct, SIGNAL(triggered()), this, SLOT(save()) );

  exitAct = new QAction( QIcon( ":/icons/exit.png" ), tr( "E&xit" ), this );
  exitAct->setShortcut( tr( "Ctrl+Q" ) );
  connect( exitAct, SIGNAL(triggered()), this, SLOT(close()) );

  customizeAct = new QAction( tr( "Customize" ), this );
  customizeAct->setShortcut( tr( "Ctrl+M" ) );
  customizeAct->setStatusTip( tr( "Customize the toolbar" ) );
  connect( customizeAct, SIGNAL(triggered()), this, SLOT(customize()) );

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
    viewerActs[i]->setCheckable( actionToolTypes[i] == CHECKABLE_TOOL );
    viewerActs[i]->setShortcut( tr( actionShortCuts[i] ) );
    viewerActs[i]->setStatusTip( tr( actionStatusTips[i] ) );
    
    viewerActs[i]->setEnabled( false );

    if ( actionToolTypes[i] == CHECKABLE_TOOL ) 
    {
      connect( viewerActs[i], SIGNAL(triggered()), this, actionSlots[i] );

      actionButtonTypes[i] == RIGHT_BUTTON ? rightButtonGroup->addAction( viewerActs[i] ) :
                                             leftButtonGroup->addAction( viewerActs[i] );
    }
  }
  // add empty action also to right group (triggering it will uncheck tools in both groups)
  rightButtonGroup->addAction( viewerActs[ACTION_EMPTY] );

  layoutAct = new QAction( QIcon( ":/icons/layout.png" ), tr( "S&creen Layout" ), this );
  layoutAct->setShortcut( tr( "Ctrl+C" ) );
  layoutAct->setStatusTip( tr( "Redisplay series and images in various layouts" ) );
  connect( layoutAct, SIGNAL(triggered()), this, SLOT(layout()) );

  replaceAct = new QAction( QIcon( ":/icons/swap.png" ), tr( "Rep&lace Viewer" ), this );
  replaceAct->setCheckable( true );
  replaceAct->setShortcut( tr( "Ctrl+L" ) );
  replaceAct->setStatusTip( tr( "Replace selected viewer (slice viewer/3D viewer)" ) );
  connect( replaceAct, SIGNAL(triggered()), this, SLOT(replace()) );
}


void m4dGUIMainWindow::createMenus ()
{
  // File Menu
  fileMenu = new QMenu( tr( "&File" ), this );
  fileMenu->addAction( searchAct );
  fileMenu->addAction( openAct );
  fileMenu->addAction( loadAct );
  fileMenu->addAction( saveAct );
  fileMenu->addSeparator();
  fileMenu->addAction( exitAct );

  menuBar()->addMenu( fileMenu );

  // ToolBars Menu
  toolBarsMenu = new QMenu( tr( "T&oolBars" ), this );
  toolBarsMenu->addAction( customizeAct );
  toolBarsMenu->addSeparator();
  toolBarsMenu->addAction( smallAct );
  toolBarsMenu->addAction( mediumAct );
  toolBarsMenu->addAction( largeAct );
  toolBarsMenu->addSeparator();
  toolBarsMenu->addAction( statusAct );

  menuBar()->addMenu( toolBarsMenu );

  // Tools Menu
  toolsMenu = new QMenu( tr( "&Tools" ), this );
  toolsMenu->addAction( layoutAct );
  toolsMenu->addSeparator();
  for ( unsigned i = 1; i < VIEWER_ACTIONS_NUMBER; i++ ) {
    toolsMenu->addAction( viewerActs[i] );
  }
  toolsMenu->addSeparator();
  toolsMenu->addAction( replaceAct );

  menuBar()->addMenu( toolsMenu );

  // View Menu
  viewMenu = new QMenu( tr( "&View" ), this );
}


void m4dGUIMainWindow::createToolBars ()
{
  searchToolBar = addToolBar( tr( "Search" ) );
  searchToolBar->addAction( searchAct );

  fileToolBar = addToolBar( tr( "File" ) );
  fileToolBar->addAction( openAct );
  fileToolBar->addAction( loadAct );
  fileToolBar->addAction( saveAct );

  layoutToolBar = addToolBar( tr( "Layout" ) );
  layoutToolBar->addAction( layoutAct );

  viewerToolBar = addToolBar( tr( "Viewer" ) );
  for ( unsigned i = 1; i < VIEWER_ACTIONS_NUMBER; i++ ) {
    viewerToolBar->addAction( viewerActs[i] );
  }

  replaceToolBar = addToolBar( tr( "Replace" ) );
  replaceToolBar->addAction( replaceAct );

  addToolBarBreak(); 

  sourcesToolBar = addToolBar( tr( "Sources" ) );
  sourcesComboBox = new QComboBox;
  sourcesComboBox->setSizeAdjustPolicy( QComboBox::AdjustToContents );
  connect( sourcesComboBox, SIGNAL(activated( int )), 
           currentViewerDesktop, SLOT(sourceSelected( int )) );
  sourcesToolBar->addWidget( sourcesComboBox );
  sourcesToolBar->hide();
}


void m4dGUIMainWindow::createStatusBar()
{
  statusBar()->showMessage( tr( "Ready" ) );
}


void m4dGUIMainWindow::delegateAction ( unsigned actionIdx, m4dGUIAbstractViewerWidget::ButtonHandler hnd )
{
  /*
  // would be nicer
  emit toolChanged( m4dGUIAbstractViewerWidget::adjust_bc,
                    m4dGUIAbstractViewerWidget::right );
  */
  m4dGUIAbstractViewerWidget::MouseButton btn;
  if ( actionButtonTypes[actionIdx] == RIGHT_BUTTON )
  {
    btn = m4dGUIAbstractViewerWidget::right;
    currentViewerDesktop->setSelectedViewerRightTool( actionIdx );
  }
  else
  {
    btn = m4dGUIAbstractViewerWidget::left;
    currentViewerDesktop->setSelectedViewerLeftTool( actionIdx );
  }

  currentViewerDesktop->getSelectedViewerWidget()->slotSetButtonHandler( hnd, btn );
}


void m4dGUIMainWindow::process ( AbstractDataSet::Ptr inputDataSet )
{
  try {

    ConnectionInterfaceTyped< AbstractImage > *conn = new ConnectionTyped< AbstractImage >;
		conn->PutDataset( inputDataSet );

		for ( unsigned i = 0; i < currentViewerDesktop->getSelectedViewerWidget()->InputPort().Size(); i++ ) {
      currentViewerDesktop->getSelectedViewerWidget()->InputPort()[i].UnPlug();
    }
		conn->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

	} 
	catch ( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}

} // namespace GUI
} // namespace M4D
