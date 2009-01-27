
#include "ManualSegmentationWidget.h"
#include "ManualSegmentationManager.h"

ManualSegmentationWidget::ManualSegmentationWidget( QWidget * parent )
	: QWidget( parent )
{
	_viewer = new M4D::Viewer::m4dGUISliceViewerWidget2( 0 );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget( _viewer );
	setLayout(mainLayout);
}
