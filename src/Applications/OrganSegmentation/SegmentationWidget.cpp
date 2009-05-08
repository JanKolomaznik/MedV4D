
#include "SegmentationWidget.h"
#include "ManualSegmentationManager.h"

typedef	M4D::GUI::GenericViewerFactory< M4D::Viewer::m4dGUISliceViewerWidget2 > SViewerFactory;

SegmentationViewerWidget::SegmentationViewerWidget( QWidget * parent )
	: M4D::GUI::m4dGUIMainViewerDesktopWidget( 1, 1, new SViewerFactory(), parent )
{
	/*_viewer = new M4D::Viewer::m4dGUISliceViewerWidget2( 0 );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget( _viewer );
	setLayout(mainLayout);*/
}

void
SegmentationViewerWidget::Activate( M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AbstractImage> *conn, M4D::Viewer::SliceViewerSpecialStateOperatorPtr specialState )
{
	setConnectionForAll( NULL );
	setConnectionForAll( conn );

	/*_viewer->setInputPort();
	_viewer->setInputPort( conn );
	_viewer->setSpecialState( specialState );
	_viewer->slotSetButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::specialState, M4D::Viewer::m4dGUIAbstractViewerWidget::left );*/
	

	M4D::Viewer::m4dGUISliceViewerWidget2	*viewer = dynamic_cast<M4D::Viewer::m4dGUISliceViewerWidget2*>( viewers[0]->viewerWidget );
	if( viewer ) {
		viewer->setSpecialState( specialState );
		viewer->slotSetButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::specialState, M4D::Viewer::m4dGUIAbstractViewerWidget::left );
	}
}
