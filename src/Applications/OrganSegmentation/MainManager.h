#ifndef MAIN_MANAGER_H
#define MAIN_MANAGER_H

#include "backendForDICOM.h"
#include "Imaging.h"

typedef M4D::Imaging::AbstractImage::Ptr	AbstractImagePtr;
typedef M4D::Imaging::Image< int16, 3 >	InputImageType;
typedef InputImageType::Ptr	InputImagePtr;
typedef M4D::Imaging::ConnectionTyped< M4D::Imaging::AbstractImage >	InImageConnection;
typedef M4D::Imaging::ConnectionTyped< InputImageType >	ImageConnectionType;

class MainManager
{
public:
	static void
	Initialize();

	static void
	Finalize();

	static void
	InitInput( M4D::Dicom::DicomObjSetPtr dicomObjSet );

	static ImageConnectionType *
	GetInputConnection()
		{ return _inConvConnection; }

	static InputImagePtr
	GetInputImage()
		{ return _inputImage = _inConvConnection->GetDatasetPtrTyped(); }
protected:
	static M4D::Dicom::DicomObjSetPtr		_inputDcmSet;
	static InputImagePtr 					_inputImage;
	static InImageConnection				*_inConnection;
	static ImageConnectionType				*_inConvConnection;
	static M4D::Imaging::PipelineContainer			_conversionPipeline;
};

#endif //MAIN_MANAGER_H


