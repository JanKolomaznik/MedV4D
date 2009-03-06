#include "iostream"
#include "Imaging/BSpline.h"

using namespace std;

using namespace M4D;
using namespace M4D::Imaging::Geometry;

typedef float CoordType;
typedef BSpline< CoordType, 2 > Curve;
typedef PointSet< CoordType, 2 > 	Points;
typedef Vector< CoordType, 2 > Coords;

void
CheckSelfIntersection( Curve &_curve )
{
	CoordInt2D seg;
	if( !FindBSplineSelfIntersection( _curve, seg ) ) return;
	
	std::cerr << seg << "\n";
	if( seg[0] + Curve::Degree >= seg[1] ) {
		_curve.RemovePoints( MOD(seg[1], _curve.Size()), MOD(seg[0] + Curve::Degree + 1, _curve.Size()) );
	}	

	unsigned inSegCount = seg[1]-(seg[0]+1);

	if( inSegCount < _curve.GetSegmentCount() - inSegCount - 2 ) {
		_curve.RemovePoints( MOD(seg[0]+2, _curve.Size()), MOD(seg[1]+2, _curve.Size()) );
	} else {
		_curve.RemovePoints( MOD(seg[1]+2, _curve.Size()), MOD(seg[0]+2, _curve.Size()) );
	}
	
}


int
main( int argc, char **argv )
{
	Curve curve;
	curve.SetCyclic();
	//Add points

	float radius = 70.0f;
	Coords center = Coords(127,127);
	int segments = 6;
	float angle = -2*PI / (float)segments;

	for( int i = 0; i < segments; ++i ) {
		Coords np = center + (radius * Coords(sin(angle*i), cos(angle*i)) );
		curve.AddPoint( np );
	}
	Coords tmp = curve[4];
	curve[4] = curve[5];
	curve[5] = tmp;
	curve.SplitSegment( 7 );
	/*
	curve.SplitSegment( 3 );
	curve[5] += Coords( 30, 0 );
	curve.SplitSegment( 4 );
	curve.SplitSegment( 3 );
	curve.SplitSegment( 5 );
	curve.SplitSegment( 7 );
	*/

	curve.Sample( 10 );

	PrintPointSet( std::cout, curve );
	std::cout << curve[0] << std::endl;
	cout << "\n\n";
	PrintCurve( std::cout, curve );
	cout << "\n\n";
	PrintCurveSegmentPoints( std::cout, curve );
	std::cout << curve.GetSamplePoints()[0] << std::endl;
	cout << "\n\n";
	
	//curve.SplitSegment( 5 );
	CheckSelfIntersection(curve);
	curve.Sample( 10 );

	PrintPointSet( std::cout, curve );
	std::cout << curve[0] << std::endl;
	cout << "\n\n";
	PrintCurve( std::cout, curve );
	cout << "\n\n";

	return 0;
}
