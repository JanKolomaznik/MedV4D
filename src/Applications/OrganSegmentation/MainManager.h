#ifndef MAIN_MANAGER_H
#define MAIN_MANAGER_H

#include "dicomConn.h"
#include "Imaging.h"

class MainManager
{
public:
	static void
	Initialize();

	static void
	Finalize();

	static void
	InitInput( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet );

	static M4D::Imaging::AbstractImageConnectionInterface *
	GetInputConnection()
		{ return _inConnection; }
protected:
	static M4D::Dicom::DcmProvider::DicomObjSetPtr		_inputDcmSet;
	static M4D::Imaging::AbstractImage::AImagePtr 		_inputImage;
	static M4D::Imaging::AbstractImageConnectionInterface	*_inConnection;
};

#endif //MAIN_MANAGER_H


