
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

void
ManualSegmentationWidget::Activate()
{
	ManualSegmentationManager::Initialize();

	_viewer->setInputPort( ManualSegmentationManager::GetInputConnection().get() );
	_viewer->setSpecialState( ManualSegmentationManager::GetSpecialState() );
	_viewer->slotSetButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::specialState, M4D::Viewer::m4dGUIAbstractViewerWidget::left );
	
}
