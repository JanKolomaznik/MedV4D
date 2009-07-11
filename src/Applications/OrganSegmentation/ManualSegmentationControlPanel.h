#ifndef MANUAL_SEGMENTATION_CONTROL_PANEL_H
#define MANUAL_SEGMENTATION_CONTROL_PANEL_H

#include <QtGui>

#include "ManualSegmentationManager.h"

class ManualSegmentationControlPanel: public QWidget
{
	Q_OBJECT;
public:
	ManualSegmentationControlPanel( ManualSegmentationManager *manager );

public slots:
	void
	PanelUpdate();

protected:
	void
	CreateWidgets();

private:

	ManualSegmentationManager	*_manager;
	QPushButton	*_deleteCurveButton;
	QAction		*_createSplineButton;
	QAction		*_editPointsButton;
};

#endif /*MANUAL_SEGMENTATION_CONTROL_PANEL_H*/
