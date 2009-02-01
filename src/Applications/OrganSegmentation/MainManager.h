#ifndef MAIN_MANAGER_H
#define MAIN_MANAGER_H

#include "dicomConn.h"
#include "Imaging.h"

typedef M4D::Imaging::AbstractImage::Ptr	AbstractImagePtr;
typedef M4D::Imaging::Image< int16, 3 >	InputImageType;
typedef InputImageType::Ptr	InputImagePtr;
typedef M4D::Imaging::AbstractImageConnectionInterface	InImageConnection;
typedef M4D::Imaging::ImageConnection< InputImageType >	ImageConnectionType;

class MainManager
{
public:
	static void
	Initialize();

	static void
	Finalize();

	static void
	InitInput( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet );

	static ImageConnectionType *
	GetInputConnection()
		{ return _inConvConnection; }

	static InputImagePtr
	GetInputImage()
		{ return _inputImage = _inConvConnection->GetImagePtr(); }
protected:
	static M4D::Dicom::DcmProvider::DicomObjSetPtr		_inputDcmSet;
	static InputImagePtr 					_inputImage;
	static InImageConnection				*_inConnection;
	static ImageConnectionType				*_inConvConnection;
	static M4D::Imaging::PipelineContainer			_conversionPipeline;
};

#endif //MAIN_MANAGER_H


