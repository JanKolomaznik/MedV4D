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
inline float32
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
inline float32
PointLinePositionPointVector( 
		const Vector< CoordType, 2 > &point,
		const Vector< CoordType, 2 > &A, 
		const Vector< CoordType, 2 > &v
	     )
{
	return (v[0])*(point[1]-A[1])
		- (v[1])*(point[0]-A[0]);
}
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/



#endif /*GEOMETRIC_ALGORITHMS*/
