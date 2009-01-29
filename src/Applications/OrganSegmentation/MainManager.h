#ifndef MAIN_MANAGER_H
#define MAIN_MANAGER_H

#include "dicomConn.h"
#include "Imaging.h"

typedef M4D::Imaging::AbstractImage	InputImageType;
typedef InputImageType::AImagePtr	InputImagePtr;
typedef boost::shared_ptr< M4D::Imaging::AbstractImageConnectionInterface >	ImageConnectionPtr;

class MainManager
{
public:
	static void
	Initialize();

	static void
	Finalize();

	static void
	InitInput( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet );

	static ImageConnectionPtr
	GetInputConnection()
		{ return _inConnection; }

	static InputImagePtr
	GetInputImage()
		{ return _inputImage; }
protected:
	static M4D::Dicom::DcmProvider::DicomObjSetPtr		_inputDcmSet;
	static InputImagePtr 					_inputImage;
	static ImageConnectionPtr				_inConnection;
};

#endif //MAIN_MANAGER_H


