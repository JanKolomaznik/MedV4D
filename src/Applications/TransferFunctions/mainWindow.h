#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/ThresholdingFilter.h"
#include "Imaging/filters/MaskMedianFilter.h"
#include "Imaging/filters/MaskSelection.h"
#include "Imaging/filters/ImageConvertor.h"
#include "GUI/widgets/m4dGUIMainViewerDesktopWidget.h"

#include <TFSliceViewerWidget.h>
#include <TFWindow.h>

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "TransferFunctions"

typedef int16	ElementType;
const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;
typedef M4D::Imaging::ThresholdingMaskFilter< ImageType > Thresholding;
typedef M4D::Imaging::MaskMedianFilter2D< Dim > Median2D;
typedef M4D::Imaging::MaskSelection< ImageType > MaskSelectionFilter;
typedef M4D::Imaging::ConnectionTyped< ImageType > InConnection;
typedef M4D::Imaging::ImageConvertor< ImageType > InImageConvertor;

class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:

	mainWindow ();

	void build();

protected:

	void process( M4D::Imaging::ADataset::Ptr inputDataSet );

	void CreatePipeline();

	TFWindow	*settings_;

	M4D::Imaging::PipelineContainer			_pipeline;
	M4D::Imaging::APipeFilter		*_filter;
	M4D::Imaging::APipeFilter		*_convertor;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_inConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_tmpConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_outConnection;

private:

	virtual void createDefaultViewerDesktop ();
};


#endif // MAIN_WINDOW_H


