#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "GUI/widgets/m4dGUIMainViewerDesktopWidget.h"
#include "GUI/widgets/components/AverageIntensitySliceViewerTexturePreparer.h"
#include "mainWindow.h"

class SettingsBox : public QWidget
{
	Q_OBJECT
public:
	static const unsigned MINIMUM_WIDTH = 400;
	static const unsigned EXECUTE_BUTTON_SPACING = 80;
	static const unsigned ROW_SPACING = 15;
	

	SettingsBox( M4D::GUI::m4dGUIMainViewerDesktopWidget * viewers, InImageRegistration ** registers, QWidget * parent );
	
	void
	SetEnabledExecButton( bool val )
		{ execButton->setEnabled( val ); }

	uint32
	GetInputNumber();

protected slots:

	void
	RegistrationType( int val );

	void
	ExecSingleFilter();

	void
	ExecAllFilters();

	void
	ExecFusion();

	void
        EndOfExecution( unsigned filterNum );

protected:
	void
	CreateWidgets();

	void
	ExecuteFilter( unsigned filterNum );

	M4D::GUI::m4dGUIMainViewerDesktopWidget *_viewers;
	InImageRegistration **_registerFilters;

	QWidget *_parent;

	QSpinBox *fusionNumber;
	QSpinBox *xRot, *yRot, *zRot, *xTrans, *yTrans, *zTrans;
	QComboBox *fusionType;
	QPushButton *clearButton;
	QPushButton *execButton;
	M4D::Viewer::AverageIntensitySliceViewerTexturePreparer< ElementType > averageTexturePreparer;
};

#endif /*_SETTINGS_BOX_H*/


