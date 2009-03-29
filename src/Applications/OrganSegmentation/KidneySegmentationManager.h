#ifndef KIDNEY_SEGMENTATION_MANAGER_H
#define KIDNEY_SEGMENTATION_MANAGER_H

#include "Imaging/Imaging.h"
#include "MainManager.h"
#include "SnakeSegmentationFilter.h"
#include "ManagerViewerSpecialState.h"

typedef M4D::Imaging::Geometry::BSpline< float32, 2 >	CurveType;
typedef CurveType::PointType				PointType;

typedef M4D::Imaging::SlicedGeometry< M4D::Imaging::Geometry::BSpline<float32,2> >	GDataSet;
typedef	M4D::Imaging::ConnectionTyped< GDataSet >	OutputGeomConnection;
typedef M4D::Imaging::SnakeSegmentationFilter< int16, SimpleVector< int16, 2 > > SegmentationFilter;
typedef M4D::Imaging::GaussianFilter2D< InputImageType >	Gaussian;
typedef M4D::Imaging::LaplaceOperator2D< InputImageType >	Laplacian;
typedef M4D::Imaging::SobelEdgeDetector< InputImageType >	Sobel;
typedef M4D::Imaging::SobelGradientOperator< InputImageType, M4D::Imaging::Image< SimpleVector< int16, 2 >, 3 > >	SobelGradient;
//typedef Laplacian	EdgeFilter;
typedef SobelGradient	EdgeFilter;

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

	GDataSet::Ptr
	GetOutputGeometry()
		{ return _outGeomConnection->GetDatasetPtrTyped(); }

	InputImagePtr
	GetInputImage()
		{ return _inputImage; }

	void
	Draw( int32 sliceNum, double zoomRate );
	void
	LeftButtonMove( Vector< float32, 2 > diff );
	void
	RightButtonMove( Vector< float32, 2 > diff ){}
	void
	RightButtonDown( Vector< float32, 2 > pos, int32 sliceNum );
	void
	LeftButtonDown( Vector< float32, 2 > pos, int32 sliceNum );

	void
	Activate( InputImageType::Ptr inImage );
public slots:
	void
	SetComputationPrecision( int value )
	{
		_computationPrecision = value;
	}

	void
	SetShapeIntensityBalance( int value )
	{
		_shapeIntensityBalance = static_cast<float32>(value) / 1000.0f;
	}

	void
	SetEdgeRegionBalance( int value )
	{
		_edgeRegionBalance = static_cast<float32>(value) / 1000.0f;
	}

	void
	SetSeparateSliceInit( bool value )
	{
		_separateSliceInit = value;
	}

	void
	StartSegmentation();

protected:
	enum SubStates { DEFINING_POLE, MOVING_POLE };

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
	Gaussian					*_gaussianFilter;

	EdgeFilter					*_edgeFilter;
	Sobel						*_sobelEdge;
	SobelGradient					*_sobelGradient;

	M4D::Imaging::CanonicalProbModel::Ptr		_probModel;

	float32						_shapeIntensityBalance;
	float32						_edgeRegionBalance;
	bool						_separateSliceInit;

	static KidneySegmentationManager		*_instance;
	bool						_wasInitialized;

	int						_actualPole;
	SubStates					_state;
	GDataSet::Ptr					_dataset;
};


#endif //KIDNEY_SEGMENTATION_MANAGER_H


