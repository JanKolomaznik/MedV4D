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
	

	SettingsBox( M4D::Imaging::AbstractPipeFilter * filter );
	
	void
	SetEnabledExecButton( bool val )
		{ execButton->setEnabled( val ); }
protected slots:

	void 
	ChangeProjPlane( int val );

	void
	ExecuteFilter();

	void
	EndOfExecution();
protected:
	void
	CreateWidgets();

	M4D::Imaging::AbstractPipeFilter *_filter;

	
	QPushButton *execButton;
};

#endif /*_SETTINGS_BOX_H*/


