#ifndef KIDNEY_SEGMENTATION_MANAGER_H
#define KIDNEY_SEGMENTATION_MANAGER_H

#include "Imaging/Imaging.h"
#include "MainManager.h"
#include "SnakeSegmentationFilter.h"
#include "ManagerViewerSpecialState.h"
#include "SegmentationManager.h"


typedef M4D::Imaging::Geometry::BSpline< float32, 2 >	CurveType;
typedef CurveType::PointType				PointType;

typedef M4D::Imaging::SlicedGeometry< M4D::Imaging::Geometry::BSpline<float32,2> >	GDataSet;
typedef	M4D::Imaging::ConnectionTyped< GDataSet >	OutputGeomConnection;
typedef M4D::Imaging::SnakeSegmentationFilter< int16, SimpleVector< int16, 2 > > SegmentationFilter;
typedef M4D::Imaging::GaussianFilter2D< InputImageType >	Gaussian;
typedef M4D::Imaging::MedianFilter2D< InputImageType >		Median;
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

class Notifier : public QObject, public M4D::Imaging::MessageReceiverInterface
{
	Q_OBJECT
public:
	Notifier() {}
	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	/*sendStyle*/, 
		M4D::Imaging::FlowDirection				/*direction*/
		)
	{
		if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
			emit Notify();
		}
	}

signals:
	void
	Notify();
protected:
};

class KidneySegmentationManager: public SegmentationManager
{
	Q_OBJECT
public:
	enum InternalState {
		DEFINING_POLE,
		POLES_SET,
		SET_SEGMENTATION_PARAMS_PREPROCESSING,
		SET_SEGMENTATION_PARAMS,
		SEGMENTATION_EXECUTED_WAITING,
		SEGMENTATION_EXECUTED_RUNNING,
		SEGMENTATION_FINISHED
	};

	static KidneySegmentationManager &
	Instance();

	std::string
	GetName()
		{ return "Kidney Segmentation"; }

	void
	Initialize();

	void
	Finalize();

	void
	RunSplineSegmentation();

	void
	RunFilters();

	M4D::Viewer::SliceViewerSpecialStateOperatorPtr
	GetSpecialState()
	{
		return _specialState;
	}
	
	InternalState
	GetInternalState()const
		{ return _state; }

	GDataSet::Ptr
	GetOutputGeometry()
		{ return _outGeomConnection->GetDatasetPtrTyped(); }

	InputImageType::Ptr
	GetInputImage()
		{ return _inputImage; }

	const ModelInfoVector &
	GetModelInfos() const
		{ return _modelInfos; }
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
	ActivateManager();

	void
	BeginManualCorrection();

	void
	PolesSet();

	void
	SetNewPoles();

	void
	FiltersFinishedSuccesfully();

	void
	SegmentationFinished();

	void
	SetComputationPrecision( int value )
	{
		_computationPrecision = value;
	}

	void
	SetDistBalance( int value )
	{
		_distBalance = static_cast<float32>(value) / 1000.0f;
	}

	void
	SetShapeBalance( int value )
	{
		_shapeBalance = static_cast<float32>(value) / 1000.0f;
	}
	
	void
	SetGeneralBalance( int value )
	{
		_generalBalance = static_cast<float32>(value) / 1000.0f;
	}

	void
	SetEdgeRegionBalance( int value )
	{
		_edgeRegionBalance = static_cast<float32>(value) / 1000.0f;
	}

	void
	SetInternalEnergyBalance( int value )
	{
		_internalEnergyBalance = static_cast<float32>(value) / 1000.0f;
	}

	void
	SetInternalEnergyGamma( int value )
	{
		_internalEnergyGamma = static_cast<float32>(value) / 1000.0f;
	}

	void
	SetSeparateSliceInit( bool value )
	{
		_separateSliceInit = value;
	}

	void
	SetModelID( int value )
	{
		_modelID = value;
	}

	void
	StartSegmentation();

signals:
	void StateUpdated();

	//ChangeInputConnection( ManagerActivationInfo info );
protected:
	InternalState	_state;
	//enum SubStates { DEFINING_POLE, MOVING_POLE };

	KidneySegmentationManager();
	
	~KidneySegmentationManager();

	void
	SetState( InternalState state );

	void
	SetReadyToSegmentationFlag( bool ready )
	{
		if( ready ) {
			_readyMutex.unlock();
		} else {
			_readyMutex.lock();
		}
		_readyToStartSegmentation = ready;
	}
	

	int						_computationPrecision;

	ImageConnectionType				*_gaussianConnection;
	ImageConnectionType				*_edgeConnection;
	OutputGeomConnection				*_outGeomConnection;
	M4D::Imaging::PipelineContainer			_container;
	InputImageType::Ptr				 	_inputImage;
	PoleDefinition					_poles[2];
	SegmentationFilter				*_segmentationFilter;
	//Gaussian					*_gaussianFilter;
	Median						*_medianFilter;

	EdgeFilter					*_edgeFilter;
	Sobel						*_sobelEdge;
	SobelGradient					*_sobelGradient;

	M4D::Imaging::CanonicalProbModel::Ptr		_probModel;

	//float32					_shapeIntensityBalance;
	float32						_distBalance;
	float32						_shapeBalance;
	float32						_generalBalance;

	float32						_edgeRegionBalance;
	float32						_internalEnergyBalance;
	float32						_internalEnergyGamma;
	bool						_separateSliceInit;

	static KidneySegmentationManager		*_instance;
	bool						_wasInitialized;

	int						_actualPole;
	//SubStates					_state;
	GDataSet::Ptr					_dataset;

	volatile bool					_readyToStartSegmentation;

	ModelInfoVector					_modelInfos;
	int						_previousModelID;
	int						_modelID;

	M4D::Multithreading::Mutex		_readyMutex;

};


#endif //KIDNEY_SEGMENTATION_MANAGER_H


