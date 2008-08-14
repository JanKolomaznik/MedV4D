#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/ThresholdingFilter.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "BoneSegmentation"

typedef int16	ElementType;
typedef M4D::Imaging::Image< ElementType, 3 > ImageType;
typedef M4D::Imaging::ThresholdingFilter< ImageType > Thresholding;
typedef M4D::Imaging::ImageConnection< ImageType > InConnection;

class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:

	mainWindow ();

protected:
	void
	process( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet );

	void
	CreatePipeline();

	M4D::Imaging::PipelineContainer	_pipeline;
	M4D::Imaging::AbstractPipeFilter	*_filter;
	M4D::Imaging::AbstractImageConnectionInterface *_inConnection;
	M4D::Imaging::AbstractImageConnectionInterface *_outConnection;

private:

};


#endif // MAIN_WINDOW_H


