
#include "ManualSegmentationManager.h"
#include "ManualSegmentationControlPanel.h"
#include "Imaging/Imaging.h"
#include "OGLDrawing.h"

using namespace M4D;
using namespace M4D::Imaging;
using namespace M4D::Imaging::Geometry;

typedef BSpline< float32, 2 >			CurveType;
typedef CurveType::PointType			PointType;

ManualSegmentationManager		*ManualSegmentationManager::_instance = NULL;
const int SAMPLE_RATE = 5;

typedef ManagerViewerSpecialState< ManualSegmentationManager >	ViewerSpecialState;

ManualSegmentationManager &
ManualSegmentationManager::Instance()
{
	if( _instance == NULL ) {
		_instance = new ManualSegmentationManager();
		//_THROW_ EInstanceUnavailable();
	}
	return *_instance;
}

void
ManualSegmentationManager::Initialize()
{
	if( _wasInitialized ) {
		Finalize();
	}
	

	//_inputImage = MainManager::Instance().GetInputImage();
	_inConnection = new ImageConnectionType( false );
	//_inConnection->PutDataset( _inputImage );

	/*int32 min = _inputImage->GetDimensionExtents(2).minimum;
	int32 max = _inputImage->GetDimensionExtents(2).maximum;
	_dataset = M4D::Imaging::DataSetFactory::CreateSlicedGeometry< M4D::Imaging::Geometry::BSpline<float32,2> >( min, max );*/

	ViewerSpecialState *sState = new ViewerSpecialState( ManualSegmentationManager::Instance() );
	_specialState = M4D::Viewer::SliceViewerSpecialStateOperatorPtr( sState );

	_controlPanel = new ManualSegmentationControlPanel( this );

	_wasInitialized = true;
}

void
ManualSegmentationManager::Finalize()
{

}

void
ManualSegmentationManager::Draw( int32 sliceNum, double zoomRate )
{
	try{
		float32 tmp;
		glGetFloatv( GL_LINE_WIDTH, &tmp );
		glLineWidth( 3.0f );

		const GDataSet::ObjectsInSlice &slice = _dataset->GetSlice( sliceNum );
		switch( _state ) {
		
		default:
			glColor3f( 0.0f, 0.0f, 1.0f );
			std::for_each( slice.begin(), slice.end(), GLDrawBSpline );
			if( sliceNum == _curveSlice && _curve ) {
				glColor3f( 1.0f, 0.0f, 0.0f );
				GLDrawBSplineCP( *_curve );
				if( SELECTED_POINT == _state ) {
					glColor3f( 0.0f, 1.0f, 0.0f );
					GLDrawPointSize( (*_curve)[ _curvePointIndex ], 3.5f );
				}
			}
		};

		glLineWidth( tmp );
	} catch (...) {

	}
}

void
ManualSegmentationManager::LeftButtonMove( Vector< float32, 2 > diff )
{
	if( SELECTED_POINT == _state ) {
		(*_curve)[ _curvePointIndex ] += diff;
		_curve->ReSample();
	}
}

void
ManualSegmentationManager::RightButtonDown( Vector< float32, 2 > pos, int32 sliceNum )
{
	switch( _state ) {
	case SELECTED:
		SetState( SELECT );
		break;
	case CREATING:
		FinishCurveCreating();
		break;
	default:
		;
	}
}

struct ClosestSplineFunctor
{
	ClosestSplineFunctor( Vector< float32, 2 > &p ) : pos( p ), minDist( TypeTraits< float32 >::Max ), counter(0), idx(-1)  {}

	void
	operator()( const CurveType &curve ) {
		float32 tmp = PolylineDistanceSquared( pos, curve.GetSamplePoints() );

		if( minDist > tmp ) {
			minDist = tmp;
			idx = counter;
		}
		++counter;
	}

	Vector< float32, 2 > pos;
	float32 minDist;
	unsigned counter;
	int32 idx;
};


void
ManualSegmentationManager::LeftButtonDown( Vector< float32, 2 > pos, int32 sliceNum )
{
	switch( _state ) {
	case SELECT:
	case SELECTED: {
			const GDataSet::ObjectsInSlice &slice = _dataset->GetSlice( sliceNum );
			ClosestSplineFunctor f = std::for_each( slice.begin(), slice.end(), ClosestSplineFunctor( pos ) );
			if( f.minDist < DISTANCE_TOLERATION_SQUARED ) {
				SetState( SELECTED );
				_curveSlice = sliceNum;
				_curveIdx = f.idx;
				_curve = &(_dataset->GetObject(sliceNum, _curveIdx));
			} else {
				SetState( SELECT );
			}
		}
		break;
	case CREATING:
		if( sliceNum != _curveSlice ) {
			SetState( CREATING );
		}
		if( _curve == NULL ) {
			PrepareNewCurve( sliceNum );
		}
		_curve->AddPoint( pos );
		_curve->ReSample();
		break;
	case SELECT_POINT:
	case SELECTED_POINT:
		{
			int32 idx = ClosestPointFromPointSet( *_curve, pos );
			PointType diff = (*_curve)[idx] - pos;
			D_PRINT( diff );
			if( (diff * diff) < DISTANCE_TOLERATION_SQUARED ) {
				_curvePointIndex = idx;
				SetState( SELECTED_POINT );
			} else {
				SetState( SELECT_POINT );
			}
		}
		break;
	default:
		;
	}
}

void
ManualSegmentationManager::PrepareNewCurve( int32 sliceNum )
{
	_curve = new CurveType();
	_curve->SetCyclic( true );
	_curve->Sample( 10 );
	_curveSlice = sliceNum;
}

void
ManualSegmentationManager::FinishCurveCreating()
{
	if( _curve == NULL ) {
		D_PRINT( "Finishing curve creation - NULL curve pointer." );
		return;
	}
	if( _curve->Size() < 4 ) {
		D_PRINT( "Finishing curve creation - deleting for low point count." );
		delete _curve;
		_curve = NULL;
		return;
	}
	D_PRINT( "Finishing curve creation - adding new curve." );
	_dataset->AddObject( _curveSlice, *_curve );
	_curve = NULL;
}

void
ManualSegmentationManager::Activate( InputImageType::Ptr inImage )
{
	int32 min = inImage->GetDimensionExtents(2).minimum;
	int32 max = inImage->GetDimensionExtents(2).maximum;
	GDataSet::Ptr dataset = M4D::Imaging::DataSetFactory::CreateSlicedGeometry< CurveType >( min, max );

	Activate( inImage, dataset );
}

void
ManualSegmentationManager::Activate( InputImageType::Ptr inImage, GDataSet::Ptr geometry )
{
	SetState( SELECT );
	_curve = NULL;
	_inputImage = inImage;
	_dataset = geometry;
	_inConnection->PutDataset( _inputImage );
}

void
ManualSegmentationManager::ActivateManagerInit( InputImageType::Ptr inImage, GDataSet::Ptr geometry )
{
	SegmentationManager::ActivateManager();

	Activate( inImage, geometry );
}

void
ManualSegmentationManager::ActivateManager()
{
	SegmentationManager::ActivateManager();

	Activate( MainManager::Instance().GetInputImage() );
}

void
ManualSegmentationManager::SetCreatingState( bool enable )
{
	if( enable ) {
		SetState( CREATING );
	} else {
		SetState( SELECT );
	}
}

void
ManualSegmentationManager::SetEditPointsState( bool enable )
{
	if( enable ) {
		SetState( SELECT_POINT );
	} else {
		SetState( SELECTED );
	}
}

void
ManualSegmentationManager::DeleteSelectedCurve()
{
	if( SELECTED == _state ) {
		_dataset->RemoveObject( _curveSlice, _curveIdx );
		_curve = NULL;
		SetState( SELECT );
	}
}

void
ManualSegmentationManager::SetState( ManualSegmentationManager::InternalState state )
{
	switch( _state ) {
	case CREATING:
		FinishCurveCreating();
		break;
	case SELECTED:
		if( state != SELECT_POINT && state != SELECTED_POINT ) {
			_curve = NULL;
		}
		break;
	default:
		;
	}
	D_PRINT( "Manual segmentation state changed from : " << _state << " to : " << state );
	_state = state;
	
	emit StateUpdated();
	emit WantsViewerUpdate();
}

