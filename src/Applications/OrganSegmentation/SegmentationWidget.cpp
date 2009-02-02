
#include "SegmentationWidget.h"
#include "ManualSegmentationManager.h"

SegmentationWidget::SegmentationWidget( QWidget * parent )
	: QWidget( parent )
{
	_viewer = new M4D::Viewer::m4dGUISliceViewerWidget2( 0 );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget( _viewer );
	setLayout(mainLayout);
}

void
SegmentationWidget::Activate( M4D::Imaging::ConnectionInterfaceTyped<AbstractImage> *conn, M4D::Viewer::SliceViewerSpecialStateOperatorPtr specialState )
{
	_viewer->setInputPort( conn );
	_viewer->setSpecialState( specialState );
	_viewer->slotSetButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::specialState, M4D::Viewer::m4dGUIAbstractViewerWidget::left );
}
