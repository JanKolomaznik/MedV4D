#ifndef MANUAL_SEGMENTATION_WIDGET_H
#define MANUAL_SEGMENTATION_WIDGET_H

#include <QtGui>
#include "MedV4D/GUI/widgets/m4dGUIMainViewerDesktopWidget.h"
#include "MedV4D/GUI/widgets/m4dGUISliceViewerWidget2.h"
#include "Imaging/Imaging.h"

class SegmentationViewerWidget: public M4D::GUI::m4dGUIMainViewerDesktopWidget
{
	Q_OBJECT;
public:
	SegmentationViewerWidget( QWidget * parent = 0 );

	void
	Activate( M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AImage> *conn, M4D::Viewer::SliceViewerSpecialStateOperatorPtr specialState );
public slots:

protected:

	//M4D::Viewer::m4dGUISliceViewerWidget2	*_viewer;
	
};


#endif // MANUAL_SEGMENTATION_WIDGET_H
