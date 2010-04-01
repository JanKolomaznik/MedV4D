#ifndef GEOMETRIC_ALGORITHMS
#define GEOMETRIC_ALGORITHMS

#include <cmath>
#include "common/Common.h"
#include "common/Vector.h"
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

enum IntersectionResult {
	ie_NO_INTERSECTION	= 0,
	ie_UNIQUE_INTERSECTION	= 1,
	ie_WHOLE_INSIDE		= 2
};

template< typename CoordType >
inline CoordType
VectorPerpDotProduct( const Vector< CoordType, 2 > &a, const Vector< CoordType, 2 > &b )
{
	return a[0] * b[1] - a[1] * b[0];
}


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
// intersect2D_2Segments(): the intersection of 2 finite 2D segments
//    Input:  two finite segments S1 and S2
//    Output: *I0 = intersect point (when it exists)
//            *I1 = endpoint of intersect segment [I0,I1] (when it exists)
//    Return: 0=disjoint (no intersect)
//            1=intersect in unique point I0
//            2=overlap in segment from I0 to I1
template< typename CoordType >
int
intersect2D_Segments( 
		const Vector< CoordType, 2 >	&pointL1A, 
		const Vector< CoordType, 2 >	&pointL1B,
		const Vector< CoordType, 2 >	&pointL2A, 
		const Vector< CoordType, 2 >	&pointL2B,
		Vector< CoordType, 2 >		&intersection
		)
		Segment S1, Segment S2, Point* I0, Point* I1 )
{
	Vector< CoordType, 2 > u( pointL1B - pointL1A );
	Vector< CoordType, 2 > v( pointL2B - pointL2A );
	Vector< CoordType, 2 > w( pointL1A - pointL2A );
	CoordType D = perp(u,v);

    // test if they are parallel (includes either being a point)
    if (fabs(D) < SMALL_NUM) {          // S1 and S2 are parallel
        if (perp(u,w) != 0 || perp(v,w) != 0) {
            return 0;                   // they are NOT collinear
        }
        // they are collinear or degenerate
        // check if they are degenerate points
        float du = dot(u,u);
        float dv = dot(v,v);
        if (du==0 && dv==0) {           // both segments are points
            if (S1.P0 != S2.P0)         // they are distinct points
                return 0;
            *I0 = S1.P0;                // they are the same point
            return 1;
        }
        if (du==0) {                    // S1 is a single point
            if (inSegment(S1.P0, S2) == 0)  // but is not in S2
                return 0;
            *I0 = S1.P0;
            return 1;
        }
        if (dv==0) {                    // S2 a single point
            if (inSegment(S2.P0, S1) == 0)  // but is not in S1
                return 0;
            *I0 = S2.P0;
            return 1;
        }
        // they are collinear segments - get overlap (or not)
        float t0, t1;                   // endpoints of S1 in eqn for S2
        Vector w2 = S1.P1 - S2.P0;
        if (v.x != 0) {
                t0 = w.x / v.x;
                t1 = w2.x / v.x;
        }
        else {
                t0 = w.y / v.y;
                t1 = w2.y / v.y;
        }
        if (t0 > t1) {                  // must have t0 smaller than t1
                float t=t0; t0=t1; t1=t;    // swap if not
        }
        if (t0 > 1 || t1 < 0) {
            return 0;     // NO overlap
        }
        t0 = t0<0? 0 : t0;              // clip to min 0
        t1 = t1>1? 1 : t1;              // clip to max 1
        if (t0 == t1) {                 // intersect is a point
            *I0 = S2.P0 + t0 * v;
            return 1;
        }

        // they overlap in a valid subsegment
        *I0 = S2.P0 + t0 * v;
        *I1 = S2.P0 + t1 * v;
        return 2;
    }

    // the segments are skew and may intersect in a point
    // get the intersect parameter for S1
    float     sI = perp(v,w) / D;
    if (sI < 0 || sI > 1)               // no intersect with S1
        return 0;

    // get the intersect parameter for S2
    float     tI = perp(u,w) / D;
    if (tI < 0 || tI > 1)               // no intersect with S2
        return 0;

    *I0 = S1.P0 + sI * u;               // compute S1 intersect point
    return 1;
}*/

/**
 * Computes relative position of point and line.
 * < 0 right from AB
 * = 0 on AB
 * > 0 left from AB
 **/
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

/**
 * Computes relative position of point and line.
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
	if( Sgn(p1) != Sgn(p2) ) {
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


template< typename CoordType >
inline IntersectionResult
LinePlaneIntersection( 
		const Vector< CoordType, 3 >	&lineA, 
		const Vector< CoordType, 3 >	&lineB,
		const Vector< CoordType, 3 >	&planePoint, 
		const Vector< CoordType, 3 >	&planeNormal,
		Vector< CoordType, 3 >		&intersection
		)
{

	Vector< CoordType, 3 > u( lineB - lineA );
	Vector< CoordType, 3 > w( lineA - planePoint );

	//only for debugging - ensure we aren't using random location later 
	D_COMMAND( intersection = Vector< CoordType, 3 >(); );

	CoordType D = planeNormal * u;
	CoordType N = -planeNormal * w;

	if ( Abs(D) < Epsilon ) {          // segment is parallel to plane
		if ( Abs(N) < Epsilon ) {                   // segment lies in plane
		    return ie_WHOLE_INSIDE;
		} else {
		    return ie_NO_INTERSECTION;                   // no intersection
		}
	}
	// they are not parallel
	// compute intersect param
	CoordType sI = N / D;
	if (sI < 0 || sI > 1) {
		return ie_NO_INTERSECTION;                       // no intersection
	}

	intersection = lineA + sI * u;                 // compute segment intersect point
	return ie_UNIQUE_INTERSECTION;
}


/** @} */

}/*namespace M4D*/



#endif /*GEOMETRIC_ALGORITHMS*/
