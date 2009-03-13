#ifndef GEOMETRIC_ALGORITHMS
#define GEOMETRIC_ALGORITHMS

#include <cmath>
#include "Common.h"
#include "Vector.h"
#include <ostream>
#include <iomanip>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file GeometricAlgorithms.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{

template< typename CoordType >
bool
LineIntersectionTest( 
		const Vector< CoordType, 2 > &pointL1A, 
		const Vector< CoordType, 2 > &pointL1B,
		const Vector< CoordType, 2 > &pointL2A, 
		const Vector< CoordType, 2 > &pointL2B
	     )
{
	Vector< CoordType, 2 > v1 = pointL1B - pointL1A;
	
	bool first = PointLinePositionPointVector( pointL2A, pointL1A, v1 ) < 0;
	bool second = PointLinePositionPointVector( pointL2B, pointL1A, v1 ) > 0;
	if( ( first && second ) || !( first || second ) ) {
		v1 = pointL2B - pointL2A;
		
		first = PointLinePositionPointVector( pointL1A, pointL2A, v1 ) < 0;
		second = PointLinePositionPointVector( pointL1B, pointL2A, v1 ) > 0;
		if( ( first && second ) || !( first || second ) ) {
			return true;
		}
	}
	return false;
}

/*
 * < 0 right from AB
 * = 0 on AB
 * > 0 left from AB
 */
template< typename CoordType >
inline CoordType
PointLinePosition2Points( 
		const Vector< CoordType, 2 > &point,
		const Vector< CoordType, 2 > &lineA, 
		const Vector< CoordType, 2 > &lineB
	     )
{
	return (lineB[0]-lineA[0])*(point[1]-lineA[1])
		- (lineB[1]-lineA[1])*(point[0]-lineA[0]);
}

/*
 * < 0 right from p
 * = 0 on p
 * > 0 left from p
 */
template< typename CoordType >
inline CoordType
PointLinePositionPointVector( 
		const Vector< CoordType, 2 > &point,
		const Vector< CoordType, 2 > &A, 
		const Vector< CoordType, 2 > &v
	     )
{
	return (v[0])*(point[1]-A[1])
		- (v[1])*(point[0]-A[0]);
}

template< typename CoordType >
inline Vector< CoordType, 2 >
PerpendicularVectorToVector( const Vector< CoordType, 2 > &v )
{
	return Vector< CoordType, 2 >( -v[1], v[0] );
}


template< typename CoordType >
inline CoordType
PointLineDistanceSquared( 
		const Vector< CoordType, 2 > &point,
		const Vector< CoordType, 2 > &A, 
		const Vector< CoordType, 2 > &v
		)
{
	Vector< CoordType, 2 > tmp = point - A;
	Vector< CoordType, 2 > norm = PerpendicularVectorToVector( v );
	CoordType sizeTmp = Sqr(tmp * norm);
	return sizeTmp / (norm * norm);
}

template< typename CoordType >
inline float32
PointLineDistance( 
		const Vector< CoordType, 2 > &point,
		const Vector< CoordType, 2 > &A, 
		const Vector< CoordType, 2 > &v
		)
{
	return sqrt( PointLineDistanceSquared( point, A, v ) );
}

template< typename CoordType >
inline CoordType
PointLineSegmentDistanceSquared( 
		const Vector< CoordType, 2 > &point,
		const Vector< CoordType, 2 > &A, 
		const Vector< CoordType, 2 > &v
		)
{
	Vector< CoordType, 2 > perp = PerpendicularVectorToVector( v );
	Vector< CoordType, 2 > B = A + v;
	CoordType p1 = PointLinePositionPointVector( point, A, perp );
	CoordType p2 = PointLinePositionPointVector( point, B, perp );
	if( Sgn(p1) == Sgn(p2) ) {
		return PointLineDistanceSquared( point, A, v );
	}
	return Min( 
			(point-A)*(point-A), 
			(point-B)*(point-B) 
		  );
}

template< typename CoordType >
inline float32
PointLineSegmentDistance( 
		const Vector< CoordType, 2 > &point,
		const Vector< CoordType, 2 > &A, 
		const Vector< CoordType, 2 > &v
		)
{
	return sqrt( PointLineSegmentDistanceSquared( point, A, v ) );
}

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/



#endif /*GEOMETRIC_ALGORITHMS*/
