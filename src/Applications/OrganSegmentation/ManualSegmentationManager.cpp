
#include "ManualSegmentationManager.h"
#include "Imaging.h"

using namespace M4D;
using namespace M4D::Imaging;
using namespace M4D::Imaging::Geometry;

typedef BSpline< float32, 2 >			CurveType;
typedef CurveType::PointType			PointType;

ManualSegmentationManager		*ManualSegmentationManager::_instance = NULL;
const int SAMPLE_RATE = 5;

static void
GLDrawPoint( const PointType &point )
{
	glVertex2f( point[0], point[1] );
}

static void
GLDrawPolyline( const CurveType::SamplePointSet &polyline )
{
	glBegin( GL_LINE_LOOP );
		std::for_each( polyline.Begin(), polyline.End(), GLDrawPoint );
	glEnd();
}

static void
GLDrawBSpline( const CurveType &spline )
{
	GLDrawPolyline( spline.GetSamplePoints() );
}

class ViewerSpecialState: public M4D::Viewer::SliceViewerSpecialStateOperator
{
public:
	enum SubStates { START_NEW_SHAPE, DEFINING_SHAPE };

	ViewerSpecialState( GDataSet::Ptr dset ): _dataset( dset ), _curve( NULL ), _state( START_NEW_SHAPE )
		{  }

	void
	Draw( M4D::Viewer::SliceViewer & viewer, int sliceNum, double zoomRate )
	{
		try{
		const GDataSet::ObjectsInSlice &slice = _dataset->GetSlice( sliceNum );
		if( _state == DEFINING_SHAPE && sliceNum == _sliceNumber && _curve ) {
			for( uint32 i = 0; i < slice.size(); ++i ) {
				if( i != _curveIdx ) {
					GLDrawBSpline( slice[i] );
				}
			}

			GLDrawBSpline( *_curve );

			float32 size;
			glGetFloatv( GL_POINT_SIZE, &size );
			glPointSize(5.0);

			glBegin( GL_POINTS );
				std::for_each( _curve->Begin(), _curve->End(), GLDrawPoint );
			glEnd();

			glPointSize(size);
		} else {
			std::for_each( slice.begin(), slice.end(), GLDrawBSpline );
		}
		} catch (...) {

		}
	}


	void 
	ButtonMethodRight( int amountH, int amountV, double zoomRate )
	{

	}
	
	void 
	ButtonMethodLeft( int amountH, int amountV, double zoomRate )
	{
		(*_curve)[ _lastPointIdx ][0] += ((float32)amountH)/zoomRate;
		(*_curve)[ _lastPointIdx ][1] += ((float32)amountV)/zoomRate;
		(*_curve).ReSample();
	}
	
	void 
	SelectMethodRight( double x, double y, int sliceNum, double zoomRate )
	{

	}
	
	void 
	SelectMethodLeft( double x, double y, int sliceNum, double zoomRate )
	{
		switch( _state ) {
		case START_NEW_SHAPE:
			_sliceNumber = sliceNum;
			_curveIdx = _dataset->AddObject( sliceNum, CurveType() ) - 1;
			_curve = &(_dataset->GetObject( sliceNum, _curveIdx ));
			_curve->SetCyclic( true );
			_curve->AddPoint( PointType( x, y ) );//*0.37/zoomRate
			_lastPointIdx = 0;
			_state = DEFINING_SHAPE;
			break;
		case DEFINING_SHAPE:
			if( sliceNum != _sliceNumber ) {
				_state = START_NEW_SHAPE;
				SelectMethodLeft( x, y, sliceNum, zoomRate );
				return;
			}
			_lastPointIdx = _curve->AddPoint( PointType( x, y ) );
			//std::cerr << (*_curve)[ _lastPointIdx ] << " ";
			_curve->Sample( SAMPLE_RATE );
			break;
		default:
			ASSERT( false );
			break;
		}
	}


	GDataSet::Ptr	_dataset;

	CurveType	*_curve;
	int32		_sliceNumber;
	uint32		_curveIdx;

	SubStates	_state;
	unsigned 	_lastPointIdx;

};

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
	

	_inputImage = MainManager::Instance().GetInputImage();
	_inConnection = new ImageConnectionType( false );
	_inConnection->PutDataset( _inputImage );

	int32 min = _inputImage->GetDimensionExtents(2).minimum;
	int32 max = _inputImage->GetDimensionExtents(2).maximum;
	_dataset = M4D::Imaging::DataSetFactory::CreateSlicedGeometry< M4D::Imaging::Geometry::BSpline<float32,2> >( min, max );

	ViewerSpecialState *sState = new ViewerSpecialState( _dataset );
	_specialState = M4D::Viewer::SliceViewerSpecialStateOperatorPtr( sState );

	_wasInitialized = true;
}

void
ManualSegmentationManager::Finalize()
{
}


