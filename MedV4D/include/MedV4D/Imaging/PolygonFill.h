#ifndef POLYGON_FILL_H
#define POLYGON_FILL_H

#include "MedV4D/Imaging/PointSet.h"
#include "MedV4D/Imaging/Polyline.h"
#include "MedV4D/Imaging/ImageRegion.h"
#include <vector>
#include <list>
#include <algorithm>

namespace M4D
{
/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file PolygonFill.h
 * @{
 **/

namespace Imaging {
namespace Algorithms {

struct IntervalRecord {
        IntervalRecord ( int32 axMin, int32 axMax, int32 ayCoordinate )
                        : xMin ( axMin ), xMax ( axMax ), yCoordinate ( ayCoordinate ) {}

        int32 xMin;
        int32 xMax;
        int32 yCoordinate;
};

struct EdgeRecord {
        EdgeRecord (
                int32	ayTop,
                float32	ax,
                int32	ady,
                float32	adxy ) :yTop ( ayTop ), x ( ax ), dy ( ady ), dxy ( adxy )	{}

        int32		yTop;

        float32		x;
        int32		dy;
        float32 	dxy;

        bool
        operator< ( const EdgeRecord &rec ) const {
                return rec.yTop < yTop || ( rec.yTop == yTop && rec.x > x );
        }
};

typedef std::vector< IntervalRecord >	IntervalRecords;
typedef std::vector< EdgeRecord >	EdgeRecords;
typedef std::list< EdgeRecord >		ActiveEdgeRecords;

template< typename CoordType >
void
PolygonFill ( const M4D::Imaging::Geometry::Polyline< Vector< CoordType, 2 > > & polygon, IntervalRecords & intervals, float32 xScale = 1.0f, float32 yScale = 1.0f );

template< typename ElementType >
void
FillRegionFromIntervals ( M4D::Imaging::ImageRegion< ElementType, 2 > &region, const IntervalRecords &intervals, ElementType value );

template< typename ElementType, typename CoordType >
void
FillRegion ( M4D::Imaging::ImageRegion< ElementType, 2 > &region, const M4D::Imaging::Geometry::Polyline< Vector< CoordType, 2 > > & polygon, ElementType value );


}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

//include implementation
#include "MedV4D/Imaging/PolygonFill.tcc"

#endif /*POLYGON_FILL_H*/
