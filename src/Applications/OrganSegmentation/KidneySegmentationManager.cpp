
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

void
LoadModelInfos( ModelInfoVector &_modelInfos )
{
	Path current = boost::filesystem::current_path<Path>();
	_modelInfos.push_back( ModelInfo( "Left kidney", current / "KidneyLeft.mdl" ) );
	_modelInfos.push_back( ModelInfo( "Right kidney", current / "KidneyRight.mdl" ) );
	_modelInfos.push_back( ModelInfo( "Right kidney - contrast", current / "KidneyContrastRight.mdl" ) );
	_modelInfos.push_back( ModelInfo( "Left kidney - contrast", current / "KidneyContrastLeft.mdl" ) );
}

KidneySegmentationManager::KidneySegmentationManager()
: _state( DEFINING_POLE ), _wasInitialized( false )
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
	
	//_probModel = CanonicalProbModel::LoadFromFile( "KidneyModel.mdl" );
	
	//Prepare list of available models	
	LoadModelInfos( _modelInfos );
	_modelID = 0;
	_previousModelID = -1;
	
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
			emit WantsViewerUpdate();
			return;
		}
		if( _actualPole == 1 ) {
			SetState( POLES_SET );
		}

		_poles[_actualPole].defined = true;
		_poles[_actualPole].slice = sliceNum;
		_poles[_actualPole].coordinates = pos;
		break;
	case POLES_SET:
	case SET_SEGMENTATION_PARAMS_PREPROCESSING:
	case SET_SEGMENTATION_PARAMS:
	case SEGMENTATION_EXECUTED_WAITING:
	case SEGMENTATION_EXECUTED_RUNNING:
	case SEGMENTATION_FINISHED:
		break;
	default:
		ASSERT( false );
		break;
	}
	emit WantsViewerUpdate();
}
//********************************************************************************************
void
KidneySegmentationManager::PolesSet()
{
	//std::cout << "Slice1 = " << _poles[0].slice << "; Slice2 = " << _poles[1].slice << "\n";
	float32 sX = _inputImage->GetDimensionExtents(0).elementExtent;
	float32 sY = _inputImage->GetDimensionExtents(1).elementExtent;
	InputImageType::PointType pom( 100, 80, 0 );
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
	for( unsigned i = 0; i < 3; ++i ) {
		minP[i] = Max( minP[i], _inputImage->GetMinimum()[i] );
		maxP[i] = Min( maxP[i], _inputImage->GetMaximum()[i] );
	}
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
	emit WantsViewerUpdate();
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
	D_PRINT( "RunFilters() executed" );

	SetReadyToSegmentationFlag( false );

	_medianFilter->ExecuteOnWhole();

	//_gaussianFilter->ExecuteOnWhole();

	//while( _gaussianFilter->IsRunning() ){ /*std::cout << ".";*/ }
}

void
KidneySegmentationManager::FiltersFinishedSuccesfully()
{
	D_PRINT( "FiltersFinishedSuccesfully() executed" );

	SetState( SET_SEGMENTATION_PARAMS );
	SetReadyToSegmentationFlag( true );
}

void
KidneySegmentationManager::SegmentationFinished()
{
	SetState( SEGMENTATION_FINISHED );
	emit WantsViewerUpdate();
}

void
KidneySegmentationManager::RunSplineSegmentation()
{
	if( _previousModelID != _modelID ) {
		try {
			//std::cout << boost::filesystem::current_path<Path>() << "\n\n";
			_probModel = CanonicalProbModel::LoadFromFile( _modelInfos[_modelID].modelFilename.filename() );
		} catch( ... ) {
			std::string tmp = TO_STRING( "Model loading problem - file not found. \"" << _modelInfos[_modelID].modelFilename << "\"" );
			emit ErrorMessageSignal( QString( tmp.data() ) );
			return;//throw;
		}
	}
	_previousModelID = _modelID;

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

	

	_segmentationFilter->SetPrecision( _computationPrecision );
	_segmentationFilter->SetEdgeRegionBalance( _edgeRegionBalance );
	_segmentationFilter->SetInternalEnergyBalance( _internalEnergyBalance );
	_segmentationFilter->SetInternalEnergyGamma( _internalEnergyGamma );
	
	//_segmentationFilter->SetShapeIntensityBalance( _shapeIntensityBalance );
	_segmentationFilter->SetDistBalance( _distBalance );
	_segmentationFilter->SetShapeBalance( _shapeBalance );
	_segmentationFilter->SetGeneralBalance( _generalBalance );

	_segmentationFilter->SetSeparateSliceInit( _separateSliceInit );

	_segmentationFilter->ExecuteOnWhole();

	while( _segmentationFilter->IsRunning() ){ /*std::cout << ".";*/ }

	SegmentationFinished();


	_dataset = _outGeomConnection->GetDatasetPtrTyped();

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
	
	D_PRINT( "Emiting signals after state change." );
	emit StateUpdated();
	//emit WantsViewerUpdate();
}
