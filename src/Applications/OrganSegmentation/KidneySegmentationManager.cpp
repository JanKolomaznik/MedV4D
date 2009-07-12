
#include "Imaging/Imaging.h"
#include <cmath>
#include "KidneySegmentationManager.h"
#include "KidneySegmentationControlPanel.h"
#include "ManualSegmentationManager.h"
#include <QtGui>
#include "OGLDrawing.h"


using namespace M4D;
using namespace M4D::Imaging;
using namespace M4D::Imaging::Geometry;

KidneySegmentationManager		*KidneySegmentationManager::_instance = NULL;


const int SAMPLE_RATE = 5;

typedef ManagerViewerSpecialState< KidneySegmentationManager >	KidneyViewerSpecialState;

KidneySegmentationManager::KidneySegmentationManager()
	: _wasInitialized( false )
{

	/*_gaussianFilter = new Gaussian();
	_gaussianFilter->SetRadius( 5 );
	_container.AddFilter( _gaussianFilter );*/
	
	_medianFilter = new Median();
	_medianFilter->SetRadius( 5 );
	_container.AddFilter( _medianFilter );

	/*_edgeFilter = new EdgeFilter();
	//_edgeFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	_edgeFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_UPDATE_FINISHED );
	_container.AddFilter( _edgeFilter );*/

	M4D::Imaging::AbstractPipeFilter *filter = new Sobel();
	filter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_UPDATE_FINISHED );
	_container.AddFilter( filter );
	_edgeFilter = new EdgeFilter();
	_edgeFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_UPDATE_FINISHED );
	_container.AddFilter( _edgeFilter );

	_segmentationFilter = new SegmentationFilter();
	_container.AddFilter( _segmentationFilter );

	/*_inConnection = (ImageConnectionType*)&(_container.MakeInputConnection( *_gaussianFilter, 0, false ) );
	_gaussianConnection = (ImageConnectionType*)&(_container.MakeConnection( *_gaussianFilter, 0, *filter, 0 ) );
		_container.MakeConnection( *_gaussianFilter, 0, *_segmentationFilter, 0 );*/

	_inConnection = (ImageConnectionType*)&(_container.MakeInputConnection( *_medianFilter, 0, false ) );
	_gaussianConnection = (ImageConnectionType*)&(_container.MakeConnection( *_medianFilter, 0, *filter, 0 ) );
		_container.MakeConnection( *_medianFilter, 0, *_segmentationFilter, 0 );

	//_edgeConnection = (ImageConnectionType*)&(_container.MakeConnection( *_edgeFilter, 0, *_segmentationFilter, 1 ) );
	ConnectionInterface &conn = _container.MakeConnection( *filter, 0, *_edgeFilter, 0 );
		_container.MakeConnection( *_edgeFilter, 0, *_segmentationFilter, 1 );

	_outGeomConnection = (OutputGeomConnection*)&(_container.MakeOutputConnection( *_segmentationFilter, 0, true ) );



	Notifier * notifier = new Notifier();

	QObject::connect( notifier, SIGNAL( Notify() ), this, SLOT( FiltersFinishedSuccesfully() ), Qt::QueuedConnection );

	conn.SetMessageHook( MessageReceiverInterface::Ptr( notifier ) );

	

}

KidneySegmentationManager::~KidneySegmentationManager()
{
	/*delete _inConnection;
	delete _outGeomConnection;*/
}

KidneySegmentationManager &
KidneySegmentationManager::Instance()
{
	if( _instance == NULL ) {
		_instance = new KidneySegmentationManager();
		//_THROW_ EInstanceUnavailable();
	}
	return *_instance;
}

void
KidneySegmentationManager::Initialize()
{
	if( _wasInitialized ) {
		Finalize();
	}

	KidneyViewerSpecialState *sState = new KidneyViewerSpecialState( KidneySegmentationManager::Instance() );
	_specialState = M4D::Viewer::SliceViewerSpecialStateOperatorPtr( sState );
	
	_probModel = CanonicalProbModel::LoadFromFile( "KidneyModel.mdl" );
	
	//_inputImage = MainManager::Instance().GetInputImage();
	//_inConnection->PutDataset( _inputImage );



	//int32 min = _inputImage->GetDimensionExtents(2).minimum;
	//int32 max = _inputImage->GetDimensionExtents(2).maximum;
	//_dataset = M4D::Imaging::DataSetFactory::CreateSlicedGeometry< M4D::Imaging::Geometry::BSpline<float32, 2> >( min, max );
	
	_controlPanel = new KidneySegmentationControlPanel( this );
	
	_wasInitialized = true;
}

void
KidneySegmentationManager::Finalize()
{
}
//********************************************************************************************
void
KidneySegmentationManager::Draw( int32 sliceNum, double zoomRate )
{
	try{
		for( int i = 0; i < 2; ++i ) {
			if( sliceNum == _poles[i].slice && _poles[i].defined ) {
				GLDrawCrossMark( _poles[i].coordinates, _poles[i].radius / zoomRate );
			}
		}
		if( _dataset ) {
			const GDataSet::ObjectsInSlice &slice = _dataset->GetSlice( sliceNum );
			float32 tmp;
			glGetFloatv( GL_LINE_WIDTH, &tmp );
			glLineWidth( 3.0f );
			std::for_each( slice.begin(), slice.end(), GLDrawBSplineCP );
			glLineWidth( tmp );
		}
	} catch (...) {

	}
}

void
KidneySegmentationManager::LeftButtonMove( Vector< float32, 2 > diff )
{
	if( _actualPole >= 2 || _actualPole < 0 ) return;

	_poles[_actualPole].coordinates += diff;
}

void
KidneySegmentationManager::RightButtonDown( Vector< float32, 2 > pos, int32 sliceNum )
{
	
}

void
KidneySegmentationManager::LeftButtonDown( Vector< float32, 2 > pos, int32 sliceNum )
{
	switch( _state ) {
	case DEFINING_POLE:
		++_actualPole;
		if( _actualPole >= 2 ) {
			SetState( POLES_SET );
			return;
		}
		if( _actualPole == 1 ) {
			SetState( POLES_SET );
		}

		_poles[_actualPole].defined = true;
		_poles[_actualPole].slice = sliceNum;
		_poles[_actualPole].coordinates = pos;
		break;
	default:
		ASSERT( false );
		break;
	}
}
//********************************************************************************************
void
KidneySegmentationManager::PolesSet()
{
	//std::cout << "Slice1 = " << _poles[0].slice << "; Slice2 = " << _poles[1].slice << "\n";
	float32 sX = _inputImage->GetDimensionExtents(0).elementExtent;
	float32 sY = _inputImage->GetDimensionExtents(1).elementExtent;
	InputImageType::PointType pom( 80, 60, 0 );
	InputImageType::PointType minP( 
				Min(_poles[0].coordinates[0]/sX,_poles[1].coordinates[0]/sX),
				Min(_poles[0].coordinates[1]/sY,_poles[1].coordinates[1]/sY),
				Min(_poles[0].slice,_poles[1].slice)
				);
	//std::cout << minP << " ";
	minP -= pom;
	InputImageType::PointType maxP( 
				Max(_poles[0].coordinates[0]/sX,_poles[1].coordinates[0]/sX),
				Max(_poles[0].coordinates[1]/sY,_poles[1].coordinates[1]/sY),
				Max(_poles[0].slice,_poles[1].slice) + 1
				);
	//std::cout << maxP << "\n";
	maxP += pom;
	_inputImage = _inputImage->GetRestrictedImage( 
			_inputImage->GetSubRegion( minP, maxP )
			);

	/*std::cout << "AAAAA = ";
	_inputImage->GetElement( minP ) = 50;
	std::cout << &(_inputImage->GetElement( minP )) << ";   ";
	_inputImage->GetRegion().GetSlice(5).GetElement( Vector<int32 ,2 >(minP[0], minP[1]) ) = 50;
	std::cout << &(_inputImage->GetRegion().GetSlice(5).GetElement( Vector<int32 ,2 >(minP[0], minP[1]) )) << "\n";*/


	_inConnection->PutDataset( _inputImage );

	SetState( SET_SEGMENTATION_PARAMS_PREPROCESSING );

	RunFilters();

	//_inConnection->PutDataset( _gaussianConnection->GetDatasetPtrTyped() );
	//_inConnection->PutDataset( _edgeConnection->GetDatasetPtrTyped() );
}

void
KidneySegmentationManager::SetNewPoles()
{
	_actualPole = -1;
	_poles[0].defined = false;	
	_poles[1].defined = false;	
	SetState( DEFINING_POLE );
}

struct SegmentationExecutionThread
{
	SegmentationExecutionThread( 
		KidneySegmentationManager	*manager 
		)
		: _manager( manager ) 
	{ /*empty*/ }

	/**
	 * Method executed by thread, which has copy of this object.
	 **/
	void
	operator()()
	{
		D_PRINT( "Entering RunSplineSegmentation()" );
		_manager->RunSplineSegmentation();
	}
private:
	KidneySegmentationManager	*_manager;

};

void
KidneySegmentationManager::StartSegmentation()
{
	/*RunSplineSegmentation();*/
	D_PRINT( "Executing thread for spline segmentation wait" );
	Multithreading::Thread thread( SegmentationExecutionThread( this ) );
}

void
KidneySegmentationManager::RunFilters()
{
	SetReadyToSegmentationFlag( false );

	_medianFilter->ExecuteOnWhole();

	//_gaussianFilter->ExecuteOnWhole();

	//while( _gaussianFilter->IsRunning() ){ /*std::cout << ".";*/ }
}

void
KidneySegmentationManager::FiltersFinishedSuccesfully()
{
	SetState( SET_SEGMENTATION_PARAMS );
	SetReadyToSegmentationFlag( true );
}

void
KidneySegmentationManager::SegmentationFinished()
{
	SetState( SEGMENTATION_FINISHED );
}

void
KidneySegmentationManager::RunSplineSegmentation()
{

	if( !_readyToStartSegmentation ) {
		D_PRINT( "RunSplineSegmentation() waiting." );
		SetState( SEGMENTATION_EXECUTED_WAITING );
		Multithreading::ScopedLock lock( _readyMutex );
		D_PRINT( "RunSplineSegmentation() waiting finished." );
	}

	SetState( SEGMENTATION_EXECUTED_RUNNING );
	//KidneyViewerSpecialState *sState = (KidneyViewerSpecialState*)(_specialState.get());

	LOG( "Poles set to : [" << _poles[0].coordinates << "]; [" << _poles[1].coordinates << "]" );
	_segmentationFilter->SetFirstPoint( _poles[0].coordinates );
	_segmentationFilter->SetFirstSlice( _poles[0].slice );
	_segmentationFilter->SetSecondPoint( _poles[1].coordinates );
	_segmentationFilter->SetSecondSlice( _poles[1].slice );
	_segmentationFilter->SetProbabilityModel( _probModel.get() );

	/*_segmentationFilter->SetInsidePoint( sState->_insidePoint );
	_segmentationFilter->SetInsidePointSlice( sState->_insidePointSlice );
	_segmentationFilter->SetOutsidePoint( sState->_outsidePoint );
	_segmentationFilter->SetOutsidePointSlice( sState->_outsidePointSlice );*/

	_segmentationFilter->SetPrecision( _computationPrecision );
	_segmentationFilter->SetEdgeRegionBalance( _edgeRegionBalance );
	_segmentationFilter->SetShapeIntensityBalance( _shapeIntensityBalance );
	_segmentationFilter->SetSeparateSliceInit( _separateSliceInit );

	_segmentationFilter->ExecuteOnWhole();

	while( _segmentationFilter->IsRunning() ){ /*std::cout << ".";*/ }

	SegmentationFinished();

	//QMessageBox::information( NULL, "Execution finished", "Filter finished its work" );



	//std::cout << "Done\n";

	_dataset = _outGeomConnection->GetDatasetPtrTyped();

	/*D_PRINT( "Go to Process results." );

	MainManager::Instance().ProcessResultDatasets( _inputImage, ptr );
	*/

	//sState->ShowGeometryDataset( ptr );
	
	
}

void
KidneySegmentationManager::Activate( InputImageType::Ptr inImage )
{
	_inputImage = inImage;
	_inConnection->PutDataset( _inputImage );

	SetNewPoles();
	_dataset = GDataSet::Ptr();
}

void
KidneySegmentationManager::ActivateManager()
{
	SegmentationManager::ActivateManager();

	Activate( MainManager::Instance().GetInputImage() );
}

void
KidneySegmentationManager::BeginManualCorrection()
{
	ManualSegmentationManager::Instance().ActivateManagerInit(
			GetInputImage(), 
			GetOutputGeometry()
			);
}

void
KidneySegmentationManager::SetState( KidneySegmentationManager::InternalState state )
{
	switch( _state ) {
	case DEFINING_POLE:
		break;
	case POLES_SET:
		break;
	case SET_SEGMENTATION_PARAMS_PREPROCESSING:
		break;
	case SET_SEGMENTATION_PARAMS:
		break;
	case SEGMENTATION_EXECUTED_WAITING:
		break;
	case SEGMENTATION_EXECUTED_RUNNING:
		break;
	case SEGMENTATION_FINISHED:
		break;
	default:
		ASSERT( false );
	}
	D_PRINT( "Kidney segmentation state changed from : " << _state << " to : " << state );
	_state = state;
	
	emit StateUpdated();
	emit WantsViewerUpdate();
}
