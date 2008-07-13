#include "GUI/m4dGUIMainViewerDesktopWidget.h"



m4dGUIMainViewerDesktopWidget::m4dGUIMainViewerDesktopWidget ( QWidget *parent )
  : QWidget( parent )
{
  QHBoxLayout *mainLayout = new QHBoxLayout;

  QSplitter *splitter = new QSplitter();

  vtkRenderWindowWidget = new m4dGUIVtkRenderWindowWidget;
  vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->sphereToRenderWindow() );
  splitter->addWidget( vtkRenderWindowWidget );

  m4dGUIVtkRenderWindowWidget *vtkInit = new m4dGUIVtkRenderWindowWidget;
  vtkInit->addRenderer( vtkRenderWindowWidget->sphereToRenderWindow() );
  splitter->addWidget( vtkInit );

  mainLayout->addWidget( splitter );

  setLayout( mainLayout ); 
}


void m4dGUIMainViewerDesktopWidget::setDesktopLayout( const int rows, const int columns )
{
  QGridLayout *mainLayout = new QGridLayout;

  QSplitter *mainSplitter = new QSplitter();
  mainSplitter->setOrientation( Qt::Vertical );

  int size = rows * columns - 1;
  m4dGUIVtkRenderWindowWidget **vtk = new m4dGUIVtkRenderWindowWidget *[size];
  for ( int i = 0; i < rows; i++ )
  {
    QSplitter *splitter = new QSplitter();
    for ( int j = 0; j < columns; j++ )
    {   
      vtk[i * columns + j] = new m4dGUIVtkRenderWindowWidget;
      vtk[i * columns + j]->addRenderer( vtkRenderWindowWidget->sphereToRenderWindow() );
      splitter->addWidget( vtk[i * columns + j] );
    }
    mainSplitter->addWidget( splitter );
  }

  mainLayout->addWidget( mainSplitter );

  delete layout();
  setLayout( mainLayout );
}