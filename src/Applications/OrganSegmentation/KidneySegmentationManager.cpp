
#include "KidneySegmentationManager.h"
#include "Imaging.h"
#include <cmath>
#include "SnakeSegmentationFilter.h"

using namespace M4D;
using namespace M4D::Imaging;
using namespace M4D::Imaging::Geometry;

ImageConnectionType *				KidneySegmentationManager::_inConnection;
M4D::Viewer::SliceViewerSpecialStateOperatorPtr KidneySegmentationManager::_specialState;
InputImagePtr				 	KidneySegmentationManager::_inputImage;
GDataSet::Ptr					KidneySegmentationManager::_dataset;
PoleDefinition					KidneySegmentationManager::_poles[2];

bool						KidneySegmentationManager::_wasInitialized = false;

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

static void
GLDrawCrossMark( PointType center, float32 radius )
{
	static const float32 cosA = cos( PI / 6.0 );
	static const float32 sinA = sin( PI / 6.0 );
	static const float32 sqrt2inv = 1.0 / sqrt( 2.0 );
	float32 pomR = radius * sqrt2inv;

	glPushMatrix();

	glTranslatef( center[0], center[1], 0.0f );
	glBegin( GL_LINES );
		glVertex2f( pomR, pomR );		
		glVertex2f( -pomR, -pomR );		
		glVertex2f( -pomR, pomR );		
		glVertex2f( pomR, -pomR );		
	glEnd();

	glBegin( GL_LINE_LOOP );
		float32 px = pomR;
		float32 py = pomR;
		for( int i = 0; i < 12; ++i ) {
			glVertex2f( px, py );
			float32 oldpx = px;
			px = px * cosA - py * sinA;
			py = oldpx * sinA + py * cosA;
		}
	glEnd();

	glPopMatrix();
}

class KidneyViewerSpecialState: public M4D::Viewer::SliceViewerSpecialStateOperator
{
public:
	enum SubStates { DEFINING_POLE, MOVING_POLE };

	KidneyViewerSpecialState( PoleDefinition * first, PoleDefinition * second ): _state( DEFINING_POLE ), _actual( -1 )
		{ _poles[0] = first; _poles[1] = second; }

	void
	Draw( M4D::Viewer::SliceViewer & viewer, int sliceNum, double zoomRate )
	{
		try{
			for( int i = 0; i < 2; ++i ) {
				if( sliceNum == _poles[i]->slice && _poles[i]->defined ) {
					GLDrawCrossMark( _poles[i]->coordinates, _poles[i]->radius / zoomRate );
				}
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
		if( _actual >= 2 || _actual < 0 ) return;

		_poles[_actual]->coordinates[0] += ((float32)amountH)/zoomRate;
		_poles[_actual]->coordinates[1] += ((float32)amountV)/zoomRate;
	}
	
	void 
	SelectMethodRight( double x, double y, int sliceNum, double zoomRate )
	{

	}
	
	void 
	SelectMethodLeft( double x, double y, int sliceNum, double zoomRate )
	{
		switch( _state ) {
		case DEFINING_POLE:
			++_actual;
			if( _actual >= 2 ) return;

			_poles[_actual]->defined = true;
			_poles[_actual]->slice = sliceNum;
			_poles[_actual]->coordinates = PointType( x, y );
			break;
		default:
			ASSERT( false );
			break;
		}
	}



	SubStates	_state;

	int32		_actual;
	PoleDefinition	*_poles[2];
};


void
KidneySegmentationManager::Initialize()
{
	if( _wasInitialized ) {
		Finalize();
	}
	
	_inputImage = MainManager::GetInputImage();
	_inConnection = new ImageConnectionType( false );
	_inConnection->PutImage( _inputImage );

	int32 min = _inputImage->GetDimensionExtents(2).minimum;
	int32 max = _inputImage->GetDimensionExtents(2).maximum;
	_dataset = M4D::Imaging::DataSetFactory::CreateSlicedGeometry< float32, M4D::Imaging::Geometry::BSpline >( min, max );

	KidneyViewerSpecialState *sState = new KidneyViewerSpecialState( &(_poles[0]), &(_poles[1]) );
	_specialState = M4D::Viewer::SliceViewerSpecialStateOperatorPtr( sState );
}

void
KidneySegmentationManager::Finalize()
{

}

void
KidneySegmentationManager::UserInputFinished()
{
	float32 sX = _inputImage->GetDimensionExtents(0).elementExtent;
	float32 sY = _inputImage->GetDimensionExtents(1).elementExtent;
	InputImageType::PointType pom( 35, 35, 0 );
	InputImageType::PointType minP( 
				Min(_poles[0].coordinates[0]/sX,_poles[1].coordinates[0]/sX),
				Min(_poles[0].coordinates[1]/sY,_poles[1].coordinates[1]/sY),
				Min(_poles[0].slice,_poles[1].slice)
				);
	minP -= pom;
	InputImageType::PointType maxP( 
				Max(_poles[0].coordinates[0]/sX,_poles[1].coordinates[0]/sX),
				Max(_poles[0].coordinates[1]/sY,_poles[1].coordinates[1]/sY),
				Max(_poles[0].slice,_poles[1].slice) + 1
				);
	maxP += pom;
	_inputImage = _inputImage->GetRestrictedImage( 
			_inputImage->GetSubRegion( minP, maxP )
			);
	_inConnection->PutImage( _inputImage );
}

