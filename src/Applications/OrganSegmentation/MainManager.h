#ifndef MAIN_MANAGER_H
#define MAIN_MANAGER_H

#include "backendForDICOM.h"
#include "Imaging.h"
#include <QtCore>
#include "SliceSplineFill.h"

typedef M4D::Imaging::AbstractImage::Ptr		AbstractImagePtr;
typedef M4D::Imaging::Image< int16, 3 >			InputImageType;
typedef M4D::Imaging::Geometry::BSpline<float32,2>	CurveType;
typedef M4D::Imaging::SlicedGeometry< CurveType >	GDataSet;

typedef InputImageType::Ptr	InputImagePtr;

typedef M4D::Imaging::ConnectionTyped< M4D::Imaging::AbstractImage >	InImageConnection;
typedef M4D::Imaging::ConnectionTyped< InputImageType >			ImageConnectionType;
typedef M4D::Imaging::ConnectionTyped< GDataSet >			GDatasetConnectionType;
typedef M4D::Imaging::ConnectionTyped< M4D::Imaging::Mask3D >		Mask3DConnectionType;

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
	InitInput( M4D::Dicom::DicomObjSetPtr dicomObjSet );

	ImageConnectionType *
	GetInputConnection()
		{ return _inConvConnection; }

	InputImagePtr
	GetInputImage()
		{ return _inputImage = _inConvConnection->GetDatasetPtrTyped(); }

	void
	ProcessResultDatasets( InputImagePtr image, GDataSet::Ptr splines );
protected:
	void
	CreateResultProcessPipeline();

	M4D::Dicom::DicomObjSetPtr		_inputDcmSet;
	InputImagePtr 					_inputImage;
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


