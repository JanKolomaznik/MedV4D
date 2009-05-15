#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "GUI/widgets/m4dGUIMainViewerDesktopWidget.h"

class SettingsBox : public QWidget
{
	Q_OBJECT
public:
	static const unsigned MINIMUM_WIDTH = 400;
	static const unsigned EXECUTE_BUTTON_SPACING = 80;
	static const unsigned ROW_SPACING = 15;
	

	SettingsBox( M4D::GUI::m4dGUIMainViewerDesktopWidget * viewers, QWidget * parent );
	
	void
	SetEnabledExecButton( bool val )
		{ execButton->setEnabled( val ); }

	uint32
	GetInputNumber();

protected slots:

	void 
	xRotValueChanged( int val );

	void 
	xTransValueChanged( int val );

	void 
	yRotValueChanged( int val );

	void 
	yTransValueChanged( int val );

	void 
	zRotValueChanged( int val );

	void 
	zTransValueChanged( int val );

	void
	RegistrationType( int val );

	void
	ExecFusion();

protected:
	void
	CreateWidgets();

	M4D::GUI::m4dGUIMainViewerDesktopWidget *_viewers;

	QWidget *_parent;

	QSpinBox *fusionNumber;
	QSpinBox *xRot, *yRot, *zRot, *xTrans, *yTrans, *zTrans;
	QComboBox *fusionType;
	QPushButton *clearButton;
	QPushButton *execButton;
};

#endif /*_SETTINGS_BOX_H*/


