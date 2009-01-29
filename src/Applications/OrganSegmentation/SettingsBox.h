#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "Imaging/filters/ThresholdingFilter.h"

class SettingsBox : public QWidget
{
	Q_OBJECT
public:
	static const unsigned MINIMUM_WIDTH = 200;
	static const unsigned EXECUTE_BUTTON_SPACING = 80;
	static const unsigned ROW_SPACING = 15;
	

	SettingsBox( QWidget * parent );
	
signals:
	void
	SetToManualSignal();	
protected slots:


protected:
	void
	CreateWidgets();

	QWidget *_parent;

	QPushButton *_manualButton;
};

#endif /*_SETTINGS_BOX_H*/


