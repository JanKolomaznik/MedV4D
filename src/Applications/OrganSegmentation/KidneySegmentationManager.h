#ifndef KIDNEY_SEGMENTATION_MANAGER_H
#define KIDNEY_SEGMENTATION_MANAGER_H

#include "Imaging.h"
#include "GUI/m4dGUISliceViewerWidget2.h"
#include "MainManager.h"
#include "SnakeSegmentationFilter.h"

typedef M4D::Imaging::Geometry::BSpline< float32, 2 >	CurveType;
typedef CurveType::PointType				PointType;

typedef M4D::Imaging::SlicedGeometry< M4D::Imaging::Geometry::BSpline<float32,2> >	GDataSet;
typedef	M4D::Imaging::ConnectionTyped< GDataSet >	OutputGeomConnection;
typedef M4D::Imaging::SnakeSegmentationFilter< int16 > SegmentationFilter;
typedef M4D::Imaging::GaussianFilter2D< InputImageType >	Gaussian;
typedef M4D::Imaging::LaplaceOperator2D< InputImageType >	Laplacian;
typedef M4D::Imaging::SobelEdgeDetector< InputImageType >	Sobel;
typedef Laplacian	EdgeFilter;

struct PoleDefinition {
	PoleDefinition(): defined( false ), radius( 10.0 ), slice( 0 )
		{}

	bool		defined;
	float32		radius;
	int32		slice;
	PointType	coordinates;
};

class KidneySegmentationManager: public QObject
{
	Q_OBJECT
public:
	static KidneySegmentationManager &
	Instance();

	void
	Initialize();

	void
	Finalize();

	void
	PolesSet();

	void
	StartSegmentation();

	void
	RunSplineSegmentation();

	void
	RunFilters();

	ImageConnectionType *
	GetInputConnection()
		{ return _inConnection; }

	M4D::Viewer::SliceViewerSpecialStateOperatorPtr
	GetSpecialState()
	{
		return _specialState;
	}

public slots:
	void
	SetComputationPrecision( int value )
	{
		_computationPrecision = value;
	}
protected:
	KidneySegmentationManager();
	
	~KidneySegmentationManager();

	int						_computationPrecision;

	ImageConnectionType				*_inConnection;
	ImageConnectionType				*_gaussianConnection;
	ImageConnectionType				*_edgeConnection;
	OutputGeomConnection				*_outGeomConnection;
	M4D::Imaging::PipelineContainer			_container;
	M4D::Viewer::SliceViewerSpecialStateOperatorPtr 	_specialState;
	InputImagePtr				 	_inputImage;
	PoleDefinition					_poles[2];
	SegmentationFilter				*_segmentationFilter;
	EdgeFilter					*_edgeFilter;
	Gaussian					*_gaussianFilter;

	static KidneySegmentationManager		*_instance;
	bool						_wasInitialized;
};


#endif //KIDNEY_SEGMENTATION_MANAGER_H


