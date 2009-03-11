#ifndef POLYGON_FILL_H
#error File PolygonFill.tcc cannot be included directly!
#else

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file PolygonFill.tcc
 * @{ 
 **/

namespace Imaging
{
namespace Algorithms
{

void
ComputeFillIntervals( EdgeRecords & edgeRecords, IntervalRecords & intervals );

template< typename CoordType >
void
PolygonFill( const M4D::Imaging::Geometry::Polyline< CoordType, 2 > & polygon, IntervalRecords & intervals, float32 xScale, float32 yScale )
{
	EdgeRecords edgeRecords;
	
	if( polygon.Size() < 3 ) return;

	D_PRINT( "Edges count = " << polygon.Size() );
	PrepareEdgeRecords( polygon, edgeRecords, xScale, yScale );

	D_PRINT( "Edge records count = " << edgeRecords.size() );
	if( edgeRecords.empty() ) return;

	ComputeFillIntervals( edgeRecords, intervals );	
}

template< typename CoordType >
void
PrepareEdgeRecords( const M4D::Imaging::Geometry::Polyline< CoordType, 2 > & polygon, EdgeRecords & edgeRecords, float32 xScale = 1.0f, float32 yScale = 1.0f )
{
	typedef typename M4D::Imaging::Geometry::Polyline< CoordType, 2 >::PointType PointType;

	PointType points[2];
	points[1] = polygon[polygon.Size()-1];
	for( uint32 i = 0; i < polygon.Size(); ++i ) {
		points[0] = points[1];
		points[1] = polygon[i];
		unsigned mY = points[0][1] > points[1][1] ? 0 : 1;
		int32 y1 = ROUND( points[mY][1] / yScale );
		int32 y2 = ROUND( points[(mY+1)%2][1] / yScale );
		if( y1 == y2 ) continue;

		float32 dxy = (points[0][0]-points[1][0]) / (points[0][1]-points[1][1]);
		dxy /= xScale * yScale;
		float32 x = points[mY][0]/xScale; //+ (points[0][1]/yScale - y1)*dxy; //TODO check
		edgeRecords.push_back( EdgeRecord(
					y1,
					x,
					y1 - y2,
					dxy
					));
	}
	std::sort( edgeRecords.begin(), edgeRecords.end() );
}

struct UpdateEdgeFunctor
{
	UpdateEdgeFunctor( IntervalRecords & recs ) : intervals( &recs ), even( true ), lastX( 0.0f )
		{}

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
		rec.x -= rec.dxy;
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

inline void
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
		activeList.sort();

		D_PRINT( "active edge count : " << activeList.size() );

		if( activeList.empty() ) break;

			
		std::for_each( activeList.begin(), activeList.end(), UpdateEdgeFunctor( intervals ) );

		activeList.erase( 
				std::remove_if( activeList.begin(), activeList.end(), RemoveEdgePredicate() ),
				activeList.end() 
				);

		--actualY;
	}
}

template< typename ElementType >
struct FillFunctor
{
	FillFunctor( M4D::Imaging::ImageRegion< ElementType, 2 > &region, ElementType value ):
		_region( region ), _value( value ) {}

	void
	operator()( const IntervalRecord &rec )
	{
		if( _region.GetMinimum( 1 ) > rec.yCoordinate || _region.GetMaximum( 1 ) <= rec.yCoordinate ) { return; }

		RasterPos pos = RasterPos( Max( rec.xMin, _region.GetMinimum(0) ), rec.yCoordinate);
		for( ; pos[0] <= Min( rec.xMax, _region.GetMaximum()[0]-1 ); ++pos[0] ) {

			_region.GetElement( pos ) = _value;
		}
	}

	M4D::Imaging::ImageRegion< ElementType, 2 >	&_region;
	ElementType					_value;
	bool						_withBorder;
};

template< typename ElementType >
void
FillRegionFromIntervals( M4D::Imaging::ImageRegion< ElementType, 2 > &region, const IntervalRecords &intervals, ElementType value )
{
	D_PRINT( "Fill intervals count = " << intervals.size() );
	std::for_each( intervals.begin(), intervals.end(), FillFunctor< ElementType >( region, value ) );
}

template< typename ElementType, typename CoordType >
void
FillRegion( M4D::Imaging::ImageRegion< ElementType, 2 > &region, const M4D::Imaging::Geometry::Polyline< CoordType, 2 > & polygon, ElementType value )
{
	IntervalRecords intervals;

	D_PRINT( "Region element extents = " << region.GetElementExtents() );
	PolygonFill( polygon, intervals, region.GetElementExtents()[0], region.GetElementExtents()[1] );

	FillRegionFromIntervals( region, intervals, value );
}


}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*POLYGON_FILL_H*/

