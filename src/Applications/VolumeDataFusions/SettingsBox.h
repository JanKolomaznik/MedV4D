#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "GUI/widgets/m4dGUIMainViewerDesktopWidget.h"
#include "GUI/widgets/components/MaxAvrMinRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MaxMedMinRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MaxAvrMinGradientRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MaxMedMinGradientRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MaxMaxMinGradientRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MaxMinMinGradientRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MultiChannelGradientRGBSliceViewerTexturePreparer.h"
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

	QSpinBox *xRot, *yRot, *zRot, *xTrans, *yTrans, *zTrans;
	QSpinBox *accuracy;
	QSpinBox *threadNum;
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
	M4D::Viewer::MaxAvrMinGradientRGBSliceViewerTexturePreparer< ElementType > maxAvrMinGradientRGBTexturePreparer;
	M4D::Viewer::MaxMedMinGradientRGBSliceViewerTexturePreparer< ElementType > maxMedMinGradientRGBTexturePreparer;
	M4D::Viewer::MaxMaxMinGradientRGBSliceViewerTexturePreparer< ElementType > maxMaxMinGradientRGBTexturePreparer;
	M4D::Viewer::MaxMinMinGradientRGBSliceViewerTexturePreparer< ElementType > maxMinMinGradientRGBTexturePreparer;
	M4D::Viewer::MultiChannelGradientRGBSliceViewerTexturePreparer< ElementType > multiChannelGradientRGBTexturePreparer;
};

#endif /*_SETTINGS_BOX_H*/


