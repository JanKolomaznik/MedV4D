#include "GUI/m4dGUIMainViewerDesktopWidget.h"

#include "GUI/m4dGUIMainWindow.h"

using namespace M4D::Imaging;
using namespace M4D::Viewer;

using namespace std;


namespace M4D {
namespace GUI {

m4dGUIMainViewerDesktopWidget::m4dGUIMainViewerDesktopWidget ( QWidget *parent )
  : QWidget( parent )
{
  // ==========================================================================

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

  // ==========================================================================

  setDesktopLayout( 1, 2 );

  selectedViewer = viewers[1];
  viewers[0]->viewerWidget->slotSetSelected( true );
}


void m4dGUIMainViewerDesktopWidget::replaceSelectedViewerWidget ( ViewerType type, 
                                                                  m4dGUIAbstractViewerWidget *replacedViewer ) 
{
  ConnectionInterface *conn = replacedViewer->getInputPort();
  unsigned idx = replacedViewer->getIndex();

  m4dGUIAbstractViewerWidget *widget = 0;

  switch ( type )
  {
    case SLICE_VIEWER:
      widget = new m4dGUISliceViewerWidget( conn, idx );
      break;

    case VTK_VIEWER:
      widget = new m4dGUIVtkViewerWidget( conn, idx );
      break;

    default:
      widget = new m4dGUISliceViewerWidget( conn, idx );
  }

  connect( widget, SIGNAL(signalSetSelected( unsigned, bool )), this, SLOT(selectedChanged( unsigned )) );
  widget->slotSetSelected( true );

  selectedViewer->type = type;
  selectedViewer->viewerWidget = widget;
  selectedViewer->checkedLeftButtonTool = selectedViewer->checkedRightButtonTool = ACTION_EMPTY;

  QLayoutItem *li = layout()->itemAt( 0 );
  QSplitter *mainSplitter = (QSplitter *)(li->widget());
  QSplitter *innerSplitter = (QSplitter *)(mainSplitter->widget( idx / layoutColumns ));

  QWidget *resizedWidget = (*widget)();
  resizedWidget->resize( resizedWidget->sizeHint() );
  
  // delete old - directly before inserting the new one 
  delete replacedViewer;
  innerSplitter->insertWidget( idx % layoutColumns, resizedWidget );
}


void m4dGUIMainViewerDesktopWidget::addSource ( ConnectionInterface *conn, const char *pipelineDescription,
                                                const char *connectionDescription )
{
  sources.push_back( conn );

  emit sourceAdded ( QString( pipelineDescription ), 
                     QString( connectionDescription ) );
}


void m4dGUIMainViewerDesktopWidget::setDesktopLayout( const unsigned rows, const unsigned columns )
{
  unsigned newSize = rows * columns;
  unsigned viewersSize = viewers.size();
  int difference = newSize - viewersSize;

  if ( difference > 0 )
  {
    for ( int i = 0; i < difference; i++ ) 
    {
      Viewer *viewer = new Viewer;
      m4dGUIAbstractViewerWidget *widget = new m4dGUISliceViewerWidget( &prodconn, viewersSize + i );
      connect( widget, SIGNAL(signalSetSelected( unsigned, bool )), this, SLOT(selectedChanged( unsigned )) );
      viewer->viewerWidget = widget;
      viewer->type = SLICE_VIEWER;
      viewer->checkedLeftButtonTool = viewer->checkedRightButtonTool = ACTION_EMPTY;
      viewers.push_back( viewer );
    }
  }
  else
  {
    if ( !viewers[newSize - 1]->viewerWidget->getSelected() ) {
      viewers[newSize - 1]->viewerWidget->slotSetSelected( true );
    }
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

  layoutRows = rows;
  layoutColumns = columns;
}


void m4dGUIMainViewerDesktopWidget::selectedChanged ( unsigned index )
{
  prevSelectedViewer = selectedViewer;
  selectedViewer = viewers[index];

  prevSelectedViewer->viewerWidget->slotSetSelected( false );

  emit propagateFeatures(); 
}

} // namespace GUI
} // namespace M4D
