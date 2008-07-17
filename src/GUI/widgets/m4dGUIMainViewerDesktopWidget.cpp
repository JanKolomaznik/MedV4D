#include "GUI/m4dGUIMainViewerDesktopWidget.h"

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

  vtkRenderWindowWidget = new m4dGUIVtkRenderWindowWidget;
  vtkRenderWindowWidget->addRenderer( vtkRenderWindowWidget->sphereToRenderWindow() );
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

  glWidget = new m4dSliceViewerWidget();
  
  splitter->addWidget( glWidget );

  // ------------------------------

  mainLayout->addWidget( splitter );

  // setLayout( mainLayout ); 
}


void m4dGUIMainViewerDesktopWidget::setDesktopLayout( const int rows, const int columns )
{
  unsigned size = rows * columns;
  int difference = size - viewers.size();

  if ( difference > 0 )
  {
    for ( unsigned i = 0; i < difference; i++ ) {
      viewers.push_back( new m4dSliceViewerWidget() );
    }
  }
  else
  {
    for ( unsigned i = size; i < viewers.size(); i++ ) {
      delete viewers[i];
    }
    viewers.resize( size );
  }

  QGridLayout *mainLayout = new QGridLayout;

  QSplitter *mainSplitter = new QSplitter();
  mainSplitter->setOrientation( Qt::Vertical );

  for ( unsigned i = 0; i < rows; i++ )
  {
    QSplitter *splitter = new QSplitter();
    for ( unsigned j = 0; j < columns; j++ )
    {   
      QWidget *widget = (*viewers[i * columns + j])();
      widget->setMinimumSize( QSize( 100, 100 ) );
      splitter->addWidget( new QGLWidget() );
    }
    mainSplitter->addWidget( splitter );
  }

  mainLayout->addWidget( mainSplitter );

  delete layout();
  setLayout( mainLayout );
}