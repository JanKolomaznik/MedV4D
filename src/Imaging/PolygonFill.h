#ifndef POLYGON_FILL_H
#define POLYGON_FILL_H

#include "Imaging/PointSet.h"
#include "Imaging/Polyline.h"
#include <vector>
#include <list>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file PolygonFill.h 
 * @{ 
 **/

namespace Imaging
{
namespace Algorithms
{

struct IntervalRecord
{
	int32 xMin;
	int32 xMax;
	int32 yCoordinate;
};

struct EdgeRecord
{
	EdgeRecord( int32 ayTop, float32 ax, int32 ady, float32 adxy ) :
		yTop( ayTop ), x( ax ), dy( ady ), dxy( adxy )	{}

	int32		yTop;

	float32		x;
	int32		dy;
	float32 	dxy;
};

struct EdgeRecordComparator
{

};

typedef std::vector< IntervalRecord >	IntervalRecords;
typedef std::vector< EdgeRecord >	EdgeRecords;
typedef std::list< EdgeRecord >		ActiveEdgeRecords;

template< typename PolygonType >
void
PolygonFill( const PolygonType & polygon, IntervalRecords & intervals )
{
	EdgeRecords edgeRecords;

	PrepareEdgeRecords( polygon, edgeRecords );

	if( edgeRecords.empty() ) return;

	ComputeFillIntervals( edgeRecords, intervals );	
}

template< typename PolygonType >
void
PrepareEdgeRecords( const PolygonType & polygon, EdgeRecords & edgeRecords )
{

}

struct UpdateEdgeFunctor
{
	UpdateEdgeFunctor( IntervalRecords & recs ) : intervals( &recs ), even( true ), lastX( 0.0f )

	void
	operator()( EdgeRecord & rec ) {
		if( even ) {
			lastX = rec.x;
		} else {
			intervals->push_back( IntervalRecord( lastX, rec.x, rec.yTop ) );
		}
		even = !even;

		--rec.yTop;
		--rec.dy;
		rec.x += rec.dxy;
	}

	IntervalRecords *intervals;
	bool	even;
	float32 lastX;
};


struct RemoveEdgePredicate
{
	bool
	operator()( const EdgeRecord & rec ) {
		return rec.dy == 0;
	}
};

template< typename PolygonType >
void
ComputeFillIntervals( EdgeRecords & edgeRecords, IntervalRecords & intervals )
{
	ActiveEdgeRecords activeList;
	unsigned nextRecord = 0;
	int32 actualY = edgeRecords[nextRecord].yTop;

	while( true ) {
		while( nextRecord < edgeRecords.size() && edgeRecords[nextRecord].yTop == actualY ) {
			activeList.push_back( edgeRecords[nextRecord] );
			++nextRecord;
		}
		activeList.Sort( EdgeRecordComparator() );

		if( activeList.empty() ) break;

	
		std::for_each( activeList.begin(), activeList.end(), UpdateEdgeFunctor( intervals ) );

		std::remove_if( activeList.begin(), activeList.end(), RemoveEdgePredicate() );

	}
}

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*POLYGON_FILL_H*/
