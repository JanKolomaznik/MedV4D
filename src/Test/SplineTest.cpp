#include "Imaging/ParametricCurve.h"
#include "iostream"

using namespace std;

using namespace M4D;
using namespace M4D::Imaging::Geometry;

typedef ParametricCurve< float, 2, BSplineBasis > Curve;
typedef Coordinates< float, 2 > Coords;

int
main( int argc, char **argv )
{
	Curve curve;
	//Add points
	Coords xstep( 10, 0 );
	Coords ystep( 0, 10 );

	Coords actualPoint( 0, 0 );
	for( unsigned i = 0; i < 10; ++i ) {
		curve.AddPoint( actualPoint );	


	}

	//Sample
	curve.Sample( 5 );

	//Output
	for( unsigned i = 0; i < curve.Size(); ++i ) {
		
		//cout <<
	}
	
	return 0;
}
