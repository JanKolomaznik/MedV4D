#include <QtGui/QtGui>

#include "m4dGUIMainWindow.h"

// just for pilot implementation
#include "../m4dPilot/renderWindowUtils.h"


m4dGUIMainWindow::m4dGUIMainWindow ()
{
  centralWidget = new QWidget;
  setCentralWidget( centralWidget );

  createVtkRenderWindowWidget();
  // just for pilot implementation
  vtkRenderWindowWidget->addRenderer( sphereToRenderWindow() );

  createActions();
  createMenus();

  QVBoxLayout *mainLayout = new QVBoxLayout;
  mainLayout->addWidget( vtkRenderWindowWidget );
  centralWidget->setLayout( mainLayout );

  setWindowTitle( tr( "m4dPilot" ) ); 
  resize( 800, 600 );
}


void m4dGUIMainWindow::open ()
{
  QString path( QFileDialog::getOpenFileName( this, tr( "Open File" ), 
                                              QDir::currentPath() ) );
  
  if ( !path.isNull() ) 
  {
    QFileInfo pathInfo( path );
    QString dirName( pathInfo.absolutePath() );

    vtkRenderWindowWidget->removeFirstRenderer();
    vtkRenderWindowWidget->addRenderer( dicomToRenderWindow( dirName.toAscii() ) );
  } 
}

void m4dGUIMainWindow::search ()
{

}


void m4dGUIMainWindow::createVtkRenderWindowWidget ()
{
  vtkRenderWindowWidget = new m4dGUIVtkRenderWindowWidget;
}


void m4dGUIMainWindow::createActions ()
{
  searchAct = new QAction( tr( "&Search" ), this );
  searchAct->setShortcut( tr( "Ctrl+S" ) );
  connect( searchAct, SIGNAL(triggered()), this, SLOT(search()) );

  openAct = new QAction( tr( "&Open..." ), this );
  openAct->setShortcut( tr( "Ctrl+O" ) );
  connect( openAct, SIGNAL(triggered()), this, SLOT(open()) );

  exitAct = new QAction( tr( "E&xit" ), this );
  exitAct->setShortcut( tr( "Ctrl+Q" ) );
  connect( exitAct, SIGNAL(triggered()), this, SLOT(close()) );
}


void m4dGUIMainWindow::createMenus ()
{
  fileMenu = new QMenu( tr( "&File" ), this );
  fileMenu->addAction( searchAct );
  fileMenu->addAction( openAct );
  fileMenu->addSeparator();
  fileMenu->addAction( exitAct );

  menuBar()->addMenu( fileMenu );
}
