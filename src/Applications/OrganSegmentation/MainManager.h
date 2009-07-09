#ifndef MAIN_MANAGER_H
#define MAIN_MANAGER_H

#include "backendForDICOM/DICOMServiceProvider.h"
#include "Imaging/Imaging.h"
#include <QtCore>
#include "SliceSplineFill.h"

#include "TypeDeclarations.h"

class MainManager;

class MainManager: public QObject
{
	Q_OBJECT;
public:
	static MainManager &
	Instance();

	void
	Initialize();

	void
	Finalize();

	void
	InitInput( M4D::Imaging::AbstractDataSet::Ptr inputDataSet );

	ImageConnectionType *
	GetInputConnection()
		{ return _inConvConnection; }

	InputImageType::Ptr
	GetInputImage()
		{ return _inputImage = _inConvConnection->GetDatasetPtrTyped(); }

	void
	ProcessResultDatasets( InputImageType::Ptr image, GDataSet::Ptr splines );
protected:
	void
	CreateResultProcessPipeline();

	M4D::Dicom::DicomObjSetPtr		_inputDcmSet;
	InputImageType::Ptr 					_inputImage;
	InImageConnection				*_inConnection;
	ImageConnectionType				*_inConvConnection;
	M4D::Imaging::PipelineContainer			_conversionPipeline;

	M4D::Imaging::PipelineContainer			_resultProcessingPipeline;
	ImageConnectionType				*_inResultImageConnection;
	GDatasetConnectionType				*_inResultGDatasetConnection;
	Mask3DConnectionType				*_resultProcessMaskConnection;

	M4D::Imaging::SliceSplineFill< float32 >			*_splineFillFilter;

	static MainManager * _instance;
};

#endif //MAIN_MANAGER_H


