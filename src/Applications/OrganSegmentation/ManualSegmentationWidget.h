#ifndef MANUAL_SEGMENTATION_WIDGET_H
#define MANUAL_SEGMENTATION_WIDGET_H

#include <QtGui>
#include "GUI/m4dGUISliceViewerWidget2.h"

class ManualSegmentationWidget: public QWidget
{
	Q_OBJECT;
public:
	ManualSegmentationWidget( QWidget * parent = 0 );
public slots:
	void
	Activate();
protected:

	M4D::Viewer::m4dGUISliceViewerWidget2	*_viewer;
	
};


#endif // MANUAL_SEGMENTATION_WIDGET_H
