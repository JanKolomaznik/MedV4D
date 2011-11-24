/**
 *  @ingroup gui
 *  @file m4dGUIMainViewerDesktopWidget.cpp
 *  @brief some brief
 */
#include "GUI/widgets/m4dGUIMainViewerDesktopWidget.h"

#include "GUI/widgets/m4dGUIMainWindow.h"

using namespace M4D::Imaging;
using namespace M4D::Viewer;

using namespace std;


namespace M4D {
namespace GUI {

m4dGUIMainViewerDesktopWidget::m4dGUIMainViewerDesktopWidget ( const unsigned rows, const unsigned columns,
                                                               ViewerFactory *viewerFactory, QWidget *parent )
  : QWidget( parent ), viewerFactory( viewerFactory ), defaultConnection( NULL )
{
  setDesktopLayout( rows, columns );

  selectedViewer = viewers[0];
  viewers[0]->viewerWidget->slotSetSelected( true );
}


m4dGUIMainViewerDesktopWidget::~m4dGUIMainViewerDesktopWidget ()
{
  delete viewerFactory;
}


m4dGUIViewerEventHandlerInterface *m4dGUIMainViewerDesktopWidget::getSelectedViewerSourceHandler () const
{
  if ( sources.empty() ) {
    return 0;
  }

	return sources[selectedViewer->sourceIdx].hnd;
}


void m4dGUIMainViewerDesktopWidget::getViewerWidgetsWithSource ( int sourceIndex, vector< m4dGUIAbstractViewerWidget * > &viewerWidgets ) const
{
  for ( unsigned i = 0; i < viewers.size(); i++ )
  {
    if ( viewers[i]->sourceIdx == sourceIndex ) {
      viewerWidgets.push_back( viewers[i]->viewerWidget );    
    }
  }
}


void m4dGUIMainViewerDesktopWidget::setDefaultConnection ( ConnectionInterface *conn )
{
	defaultConnection = conn;
}


void m4dGUIMainViewerDesktopWidget::setConnectionForAll ( ConnectionInterface *conn )
{
  for ( unsigned i = 0; i < viewers.size(); ++i ) {
		viewers[i]->viewerWidget->setInputPort( conn );
	}
}


void m4dGUIMainViewerDesktopWidget::setViewerEventHandlerForSelected ( m4dGUIViewerEventHandlerInterface *eventHandler )
{
  selectedViewer->viewerWidget->setViewerEventHandler( eventHandler );
}


void m4dGUIMainViewerDesktopWidget::setViewerEventHandlerForAll ( m4dGUIViewerEventHandlerInterface *eventHandler )
{
  for ( unsigned i = 0; i < viewers.size(); ++i ) {
		viewers[i]->viewerWidget->setViewerEventHandler( eventHandler );
	}
}


void m4dGUIMainViewerDesktopWidget::replaceSelectedViewerWidget ( ViewerFactory *viewerFactory, 
                                                                  m4dGUIAbstractViewerWidget *replacedViewer ) 
{
  ConnectionInterface *conn = replacedViewer->getInputPort();
  unsigned idx = replacedViewer->getIndex();
  list< string > leftOverlayInfo  = replacedViewer->getLeftSideTextData();
  list< string > rightOverlayInfo = replacedViewer->getRightSideTextData();

  m4dGUIAbstractViewerWidget *widget = viewerFactory->newViewer( conn, idx );

  widget->setLeftSideTextData( leftOverlayInfo ); 
  widget->setRightSideTextData( rightOverlayInfo ); 

  connect( widget, SIGNAL(signalSetSelected( unsigned, bool )), this, SLOT(selectedChanged( unsigned )) );
  widget->slotSetSelected( true );

  selectedViewer->viewerWidget = widget;
  selectedViewer->checkedLeftButtonTool = selectedViewer->checkedRightButtonTool = ACTION_EMPTY;

  QLayoutItem *li = layout()->itemAt( 0 );
  QSplitter *mainSplitter = (QSplitter *)(li->widget());
  QSplitter *innerSplitter = (QSplitter *)(mainSplitter->widget( idx / layoutColumns ));

  QWidget *resizedWidget = (*widget)();
  resizedWidget->resize( resizedWidget->sizeHint() );

  QByteArray state = innerSplitter->saveState();

  // delete old - directly before inserting the new one 
  delete replacedViewer;
  innerSplitter->insertWidget( idx % layoutColumns, resizedWidget );

  innerSplitter->restoreState( state );
}


void m4dGUIMainViewerDesktopWidget::addSource ( vector< ConnectionInterface * > &conn, 
                                                m4dGUIViewerEventHandlerInterface *viewerEventHandler )
{
  sources.push_back( Source( conn, viewerEventHandler ) );
}


void m4dGUIMainViewerDesktopWidget::addSource ( ConnectionInterface *conn, 
                                                m4dGUIViewerEventHandlerInterface *viewerEventHandler )
{
  vector< ConnectionInterface * > connections;
  connections.push_back( conn );
  
  sources.push_back( Source( connections, viewerEventHandler ) );
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

      m4dGUIAbstractViewerWidget *widget = viewerFactory->newViewer( viewersSize + i );
	    if( defaultConnection ) {
		    widget->setInputPort( defaultConnection );
	    }

      connect( widget, SIGNAL(signalSetSelected( unsigned, bool )), this, SLOT(selectedChanged( unsigned )) );
      
      viewer->viewerWidget = widget;
      viewer->checkedLeftButtonTool = viewer->checkedRightButtonTool = ACTION_EMPTY;
      viewer->sourceIdx = 0;

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

  if ( prevSelectedViewer != selectedViewer ) {
    prevSelectedViewer->viewerWidget->slotSetSelected( false );
  }

  emit propagateFeatures( prevSelectedViewer->viewerWidget ); 
}


void m4dGUIMainViewerDesktopWidget::sourceSelected ( int index )
{ 
  selectedViewer->sourceIdx = index;

  for ( unsigned i = 0; i < selectedViewer->viewerWidget->InputPort().Size(); i++ ) {
    selectedViewer->viewerWidget->InputPort()[i].UnPlug();
  }
  for ( unsigned i = 0; i < sources[index].conn.size(); i++ ) {
    sources[index].conn[i]->ConnectConsumer( selectedViewer->viewerWidget->InputPort()[i] );
  }

  selectedViewer->viewerWidget->setViewerEventHandler( sources[index].hnd );

  emit sourceChanged(); 
}


void m4dGUIMainViewerDesktopWidget::UpdateViewers()
{
	for ( unsigned i = 0; i < viewers.size(); ++i ) {
		viewers[i]->viewerWidget->updateViewer();
	}
}

} // namespace GUI
} // namespace M4D
