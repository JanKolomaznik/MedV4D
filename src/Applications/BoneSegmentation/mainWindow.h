#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "BoneSegmentation"


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
	M4D::Imaging::AbstractImageConnection *_inConnection;
	M4D::Imaging::AbstractImageConnection *_outConnection;

private:

};


#endif // MAIN_WINDOW_H


