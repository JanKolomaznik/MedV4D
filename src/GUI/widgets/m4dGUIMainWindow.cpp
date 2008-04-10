#include <QtGui/QtGui>

#include "m4dGUIMainWindow.h"


m4dGUIMainWindow::m4dGUIMainWindow ()
{
  centralWidget = new QWidget;
  setCentralWidget( centralWidget );

  createVtkRenderWindowWidget();

  createActions();
  createMenus();

  QVBoxLayout *mainLayout = new QVBoxLayout;
  mainLayout->addWidget( vtkRenderWindowWidget );
  centralWidget->setLayout( mainLayout );

  setWindowTitle( tr( "m4dPilot" ) ); 
  resize( minimumSizeHint() );
}


void m4dGUIMainWindow::open ()
{
  QString fileName;
  fileName = QFileDialog::getOpenFileName( this, tr( "Open File" ), QDir::currentPath() );
}


void m4dGUIMainWindow::createVtkRenderWindowWidget ()
{
  vtkRenderWindowWidget = new m4dGUIVtkRenderWindowWidget;
}


void m4dGUIMainWindow::createActions ()
{
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
  fileMenu->addAction( openAct );
  fileMenu->addSeparator();
  fileMenu->addAction( exitAct );

  menuBar()->addMenu( fileMenu );
}
