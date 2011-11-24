#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "Imaging/interpolators/nearestNeighbor.h"
#include "Imaging/interpolators/linear.h"
#include "MedV4D/GUI/widgets/m4dGUIMainViewerDesktopWidget.h"
#include "MedV4D/GUI/widgets/components/SobelOperatorSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/StandardDeviationSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MaxAvrMinRGBSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MaxMedMinRGBSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MaxAvrMinGradientRGBSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MaxMedMinGradientRGBSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MaxMaxMinGradientRGBSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MaxMinMinGradientRGBSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MultiChannelGradientRGBSliceViewerTexturePreparer.h"
#include "mainWindow.h"

class SettingsBox : public QWidget
{
	Q_OBJECT
public:
	static const unsigned MINIMUM_WIDTH = 400;
	static const unsigned EXECUTE_BUTTON_SPACING = 80;
	static const unsigned ROW_SPACING = 15;
	

	/**
	 * Constructor
	 *  @param viewers pointer to the viewer desktop widget
	 *  @param registers pointer to the array of register filters
	 *  @param parent pointer to the parent widget
	 */
	SettingsBox( M4D::GUI::m4dGUIMainViewerDesktopWidget * viewers, M4D::Imaging::APipeFilter ** registers, QWidget * parent );

	/**
	 * Set whether the execute button is enabled
	 *  @param val true if the execute button should be enabled, false otherwise
	 */
	void
	SetEnabledExecButton( bool val )
		{ execButton->setEnabled( val ); }

	/**
	 * Get the number of the currently selected viewer's input port
	 */
	uint32
	GetInputNumber();

	/**
	 * Execute a filter with the given number
	 *  @param filterNum the number of the filter to be executed
	 */
	void
	ExecuteFilter( unsigned filterNum );

protected slots:

	/**
	 * Set the registration's type
	 *  @param val the registration's type
	 */
	void
	RegistrationType( int val );

	/**
	 * Set the fusion's type
	 *  @param val the fusion's type
	 */
	void
	FusionType( int val );

	/**
	 * Execute a single filter
	 */
	void
	ExecSingleFilter();

	/**
	 * Execute all filters
	 */
	void
	ExecAllFilters();

	/**
	 * Stop a single filter
	 */
	void
	StopSingleFilter();

	/**
	 * Stop all filters
	 */
	void
	StopAllFilters();

	/**
	 * Show fusion of inputs in the currently selected viewer
	 */
	void
	ExecFusion();

	/**
	 * Execute when the execution of a filter is finished
	 *  @param filterNum the number of the registration filter that has completed
	 */
	void
        EndOfExecution( unsigned filterNum );

protected:

	/**
	 * Create the widgets
	 */
	void
	CreateWidgets();

	// viewer desktop widget
	M4D::GUI::m4dGUIMainViewerDesktopWidget *_viewers;

	// image registration filter
	M4D::Imaging::APipeFilter **_registerFilters;

	// parent widget
	QWidget *_parent;

	// rotation and translation setters
	QSpinBox *xRot, *yRot, *zRot, *xTrans, *yTrans, *zTrans;

	// brightness and contrast adjustaers for RGB channels in colored fusion
	QSpinBox *bRed, *bGreen, *bBlue, *cRed, *cGreen, *cBlue;

	// accuracy setter
	QSpinBox *accuracy;

	// number of threads setter
	QSpinBox *threadNum;

	// radius of SD fusion
	QSpinBox *SDradius;

	// fusion type selector
	QComboBox *fusionType;

	// interpolator type selector
	QComboBox *interpolatorType;

	// clear selected dataset button
	QPushButton *clearButton;

	// execute button
	QPushButton *execButton;

	// slice viewer texture preparers for each type of fusion
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
	M4D::Viewer::SobelOperatorSliceViewerTexturePreparer< ElementType > sobelOperatorTexturePreparer;
	M4D::Viewer::StandardDeviationSliceViewerTexturePreparer< ElementType > standardDeviationTexturePreparer;

	// interpolators for different types of interpolation
	M4D::Imaging::NearestNeighborInterpolator< ImageType >			nearestNeighborInterpolator[ SLICEVIEWER_INPUT_NUMBER ];
	M4D::Imaging::LinearInterpolator< ImageType >				linearInterpolator[ SLICEVIEWER_INPUT_NUMBER ];

};

#endif /*_SETTINGS_BOX_H*/


