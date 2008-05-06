#include "m4dGUIMainWindow.h"

#include <QtGui>


m4dGUIMainWindow::m4dGUIMainWindow ()
{
  Q_INIT_RESOURCE( m4dGUIMainWindow );

  createStudyManagerDialog();

  centralWidget = new QWidget;
  setCentralWidget( centralWidget );

  createVtkRenderWindowWidget();
  // just for pilot implementation
  //vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->sphereToRenderWindow() );
  vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->imageDataToRenderWindow() );

  createActions();
  createMenus();
  createToolBars();
  createStatusBar();

  QVBoxLayout *mainLayout = new QVBoxLayout;
  mainLayout->addWidget( vtkRenderWindowWidget );
  centralWidget->setLayout( mainLayout );

  setWindowTitle( tr( "m4dPilot" ) ); 
  setWindowIcon( QIcon( ":/icons/app.png" ) );
  resize( 800, 600 );
}


void m4dGUIMainWindow::open ()
{
  QString path( QFileDialog::getOpenFileName( this, tr( "Open File" ), 
                                              QDir::currentPath(), "*.dcm" ) );
  
  if ( !path.isNull() ) 
  {
    QFileInfo pathInfo( path );
    QString dirName( pathInfo.absolutePath() );

    vtkRenderWindowWidget->removeFirstRenderer();
    vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->dicomToRenderWindow( dirName.toAscii() ) );
  } 
}


void m4dGUIMainWindow::search ()
{
  studyManagerDialog->show();
}


void m4dGUIMainWindow::createStudyManagerDialog ()
{
  m4dGUIStudyManagerWidget *studyManagerWidget = new m4dGUIStudyManagerWidget;

  QVBoxLayout *dialogLayout = new QVBoxLayout;
  dialogLayout->addWidget( studyManagerWidget );

  studyManagerDialog = new QDialog( this );
  studyManagerDialog->setWindowTitle( tr( "Study Manager" ) );
  studyManagerDialog->setWindowIcon( QIcon( ":/icons/search.png" ) );
  studyManagerDialog->setLayout( dialogLayout );
}


void m4dGUIMainWindow::createVtkRenderWindowWidget ()
{
  vtkRenderWindowWidget = new m4dGUIVtkRenderWindowWidget;
}


void m4dGUIMainWindow::createActions ()
{
  searchAct = new QAction( QIcon( ":/icons/search.png" ), tr( "&Search" ), this );
  searchAct->setShortcut( tr( "Ctrl+S" ) );
  searchAct->setStatusTip( tr( "Search" ) );
  connect( searchAct, SIGNAL(triggered()), this, SLOT(search()) );

  openAct = new QAction( QIcon( ":/icons/open.png" ), tr( "&Open..." ), this );
  openAct->setShortcut( tr( "Ctrl+O" ) );
  openAct->setStatusTip( tr( "Open an existing document" ) );
  connect( openAct, SIGNAL(triggered()), this, SLOT(open()) );

  saveAct = new QAction( QIcon( ":/icons/save.png" ), tr( "S&ave" ), this );
  saveAct->setShortcut( tr( "Ctrl+A" ) );
  // saveAct->setStatusTip( tr( "Save the document to disk" ) );
  // connect( saveAct, SIGNAL(triggered()), this, SLOT(save()) );
  saveAct->setEnabled( false );

  exitAct = new QAction( QIcon( ":/icons/exit.png" ), tr( "E&xit" ), this );
  exitAct->setShortcut( tr( "Ctrl+Q" ) );
  connect( exitAct, SIGNAL(triggered()), this, SLOT(close()) );
}


void m4dGUIMainWindow::createMenus ()
{
  fileMenu = new QMenu( tr( "&File" ), this );
  fileMenu->addAction( searchAct );
  fileMenu->addAction( openAct );
  fileMenu->addAction( saveAct );
  fileMenu->addSeparator();
  fileMenu->addAction( exitAct );

  menuBar()->addMenu( fileMenu );
}


void m4dGUIMainWindow::createToolBars ()
{
  fileToolBar = addToolBar( tr( "File" ) );
  fileToolBar->addAction( openAct );
  fileToolBar->addAction( saveAct );

  searchToolBar = addToolBar( tr( "Search" ) );
  searchToolBar->addAction( searchAct );
}


void m4dGUIMainWindow::createStatusBar()
{
  statusBar()->showMessage( tr( "Ready" ) );
}
