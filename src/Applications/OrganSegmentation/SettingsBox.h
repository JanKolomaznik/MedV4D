#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "Imaging/filters/ThresholdingFilter.h"
#include "SegmentationTypes.h"


class SettingsBox : public QStackedWidget
{
	Q_OBJECT
public:
	static const unsigned MINIMUM_WIDTH = 200;
	static const unsigned EXECUTE_BUTTON_SPACING = 80;
	static const unsigned ROW_SPACING = 15;
	

	SettingsBox( QWidget * parent );

public slots:
	/*void
	ManualSegmentationUpdate();*/

	void
	SetToDefault();

signals:
	
	/*void
	SetSegmentationSignal( uint32 type );*/

protected slots:

	/*void
	SetToManualSegmentation();

	void
	SetToKidneySegmentation();

	void
	PolesSetKidneySegm();

	void
	StartSegmentationKidneySegm();

	void
	BeginManualCorrection();

	void
	ProcessResultsManual();*/
protected:

	void
	InitializeSementationGUI();

	/*void
	CreateWidgets();
	
	void
	CreateManualGUI();*/

	QWidget *_parent;

	QWidget *_mainSettings;

	/*QWidget *_manualSegmSettings;
	QWidget *_kidneySegmSettings;*/

};

#endif /*_SETTINGS_BOX_H*/


