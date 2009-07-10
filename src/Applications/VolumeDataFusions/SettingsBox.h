#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "GUI/widgets/m4dGUIMainViewerDesktopWidget.h"
#include "GUI/widgets/components/MaxAvrMinRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MaxMedMinRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MultiChannelRGBSliceViewerTexturePreparer.h"
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
	StopSingleFilter();

	void
	StopAllFilters();

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
	QSpinBox *accuracy;
	QComboBox *fusionType;
	QPushButton *clearButton;
	QPushButton *execButton;
	M4D::Viewer::MultiChannelRGBSliceViewerTexturePreparer< ElementType > multiChannelRGBTexturePreparer;
	M4D::Viewer::MaximumIntensitySliceViewerTexturePreparer< ElementType > maximumIntensityTexturePreparer;
	M4D::Viewer::MinimumIntensitySliceViewerTexturePreparer< ElementType > minimumIntensityTexturePreparer;
	M4D::Viewer::AverageIntensitySliceViewerTexturePreparer< ElementType > averageIntensityTexturePreparer;
	M4D::Viewer::MedianIntensitySliceViewerTexturePreparer< ElementType > medianIntensityTexturePreparer;
	M4D::Viewer::MaxAvrMinRGBSliceViewerTexturePreparer< ElementType > maxAvrMinRGBTexturePreparer;
	M4D::Viewer::MaxMedMinRGBSliceViewerTexturePreparer< ElementType > maxMedMinRGBTexturePreparer;
};

#endif /*_SETTINGS_BOX_H*/


