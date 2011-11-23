#ifndef KIDNEY_SEGMENTATION_CONTROL_PANEL_H
#define KIDNEY_SEGMENTATION_CONTROL_PANEL_H

#include <QtGui>
#include "KidneySegmentationManager.h"

class KidneySegmentationControlPanel: public QWidget
{
	Q_OBJECT;
public:
	KidneySegmentationControlPanel( KidneySegmentationManager *manager );

public slots:
	void
	PanelUpdate();

protected:
	void
	CreateWidgets();

	KidneySegmentationManager	*_manager;

	QFrame				*_polesButtons;
	QFrame				*_segmentationSettings;
	QFrame				*_resultButtons;

	QComboBox			*_modelsCombo;
	QLabel				*_infoBox;
};

#endif /*KIDNEY_SEGMENTATION_CONTROL_PANEL_H*/
