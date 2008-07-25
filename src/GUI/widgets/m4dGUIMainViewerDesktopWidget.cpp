#include "GUI/m4dGUIMainViewerDesktopWidget.h"

#include "GUI/m4dGUIMainWindow.h"

#include "Common.h"
#include "Imaging/ExampleImageFilters.h"
#include "Imaging/DefaultConnection.h"
#include "Imaging/ImageFactory.h"

using namespace M4D::Imaging;
using namespace M4D::Viewer;

// TESTING viewer
typedef Image< uint32, 3 > Image3DType;
typedef ImageConnectionSimple< Image3DType > ProducerConn;


m4dGUIMainViewerDesktopWidget::m4dGUIMainViewerDesktopWidget ( QWidget *parent )
  : QWidget( parent )
{
  QHBoxLayout *mainLayout = new QHBoxLayout;

  QSplitter *splitter = new QSplitter();

  // TODO
  // vtkRenderWindowWidget = new m4dGUIVtkViewerWidget;
  // vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->sphereToRenderWindow() );
  vtkRenderWindowWidget = new m4dGUIVtkViewerWidget( 0 );
  splitter->addWidget( vtkRenderWindowWidget );

  // TESTING viewer
  // ------------------------------

	inputImage = ImageFactory::CreateEmptyImage3DTyped< uint32 >( 512, 512, 50 );

	size_t i, j, k;
	uint8* p;
	for ( i = inputImage->GetDimensionExtents( 0 ).minimum; i < inputImage->GetDimensionExtents( 2 ).maximum; ++i )
		for ( j = inputImage->GetDimensionExtents( 1 ).minimum; j < inputImage->GetDimensionExtents( 0 ).maximum; ++j )
			for ( k = inputImage->GetDimensionExtents( 2 ).minimum; k < inputImage->GetDimensionExtents( 1 ).maximum; ++k )
			{
				p = (uint8*) &inputImage->GetElement( j, k, i );// = ( i * j * k ) % 32000;
				p[0] = i * j % 256;
				p[1] = j * k % 256;
				p[2] = i * k % 256;
				p[3] = 0;
			}

	prodconn.PutImage( inputImage );

  glWidget = new m4dGUISliceViewerWidget( 1 );
  
  splitter->addWidget( glWidget );

  // ------------------------------

  mainLayout->addWidget( splitter );

  // setLayout( mainLayout ); 
  setDesktopLayout( 1, 2 );

  selectedViewer = viewers[1];
  viewers[0]->viewerWidget->slotSetSelected( true );
}


void m4dGUIMainViewerDesktopWidget::setDesktopLayout( const int rows, const int columns )
{
  unsigned newSize = rows * columns;
  unsigned viewersSize = viewers.size();
  int difference = newSize - viewersSize;

  if ( difference > 0 )
  {
    for ( unsigned i = 0; i < difference; i++ ) 
    {
      Viewer *viewer = new Viewer;
      m4dGUISliceViewerWidget *widget = new m4dGUISliceViewerWidget( prodconn, viewersSize + i );
      connect( (m4dGUIAbstractViewerWidget *)widget, SIGNAL(signalSetSelected( unsigned, bool )), this, SLOT(selectedChanged( unsigned )) );
      viewer->viewerWidget = widget;
      viewer->checkedLeftButtonTool = ACTION_PAN;
      viewer->checkedRightButtonTool = ACTION_WINDOW_LEVEL;
      viewers.push_back( viewer );
    }
  }
  else
  {
    viewers[newSize - 1]->viewerWidget->slotSetSelected( true );
    for ( unsigned i = newSize; i < viewersSize; i++ ) {
      delete viewers[i];
    }
    viewers.resize( newSize );
  }

  QGridLayout *mainLayout = new QGridLayout;

  QSplitter *mainSplitter = new QSplitter();
  mainSplitter->setOrientation( Qt::Vertical );

  for ( unsigned i = 0; i < rows; i++ )
  {
    QSplitter *splitter = new QSplitter();
    for ( unsigned j = 0; j < columns; j++ )
    {   
      QWidget *widget = (*viewers[i * columns + j]->viewerWidget)();
      widget->resize( widget->sizeHint() );
      splitter->addWidget( widget );
    }
    mainSplitter->addWidget( splitter );
  }

  mainLayout->addWidget( mainSplitter );

  delete layout();
  setLayout( mainLayout );
}


void m4dGUIMainViewerDesktopWidget::selectedChanged ( unsigned index )
{
  selectedViewer->viewerWidget->slotSetSelected( false );
  prevSelectedViewer = selectedViewer;
  selectedViewer = viewers[index];

  emit propagateFeatures(); 
}
