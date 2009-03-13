#ifndef POLYLINE_H
#define POLYLINE_H

#include "Imaging/PointSet.h"
#include "Imaging/GeometricAlgorithms.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Polyline.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{

template < typename CoordType, unsigned Dim >
class Polyline: public PointSet< CoordType, Dim >
{
public:
	typedef Vector< CoordType, Dim > 	PointType;
	typedef CoordType			Type;
	static const unsigned Dimension	= Dim;		
	
	Polyline(): _cyclic( false ) {}

	void
	SetCyclic( bool cyclic = true )
		{ _cyclic = cyclic; }

	bool
	Cyclic() const
		{ return _cyclic; }
protected:
	bool 			_cyclic;

};

template< typename CoordType >
CoordType
PolylineDistanceSquared( const Vector< CoordType, 2 > &pos, const Polyline< CoordType, 2 > &polyline )
{
	CoordType dist = PointLineSegmentDistanceSquared( pos, polyline[0] , polyline[1] - polyline[0] );
	for( unsigned i = 2; i < polyline.Size(); ++i ) {
		dist = Min( dist, 
			PointLineSegmentDistanceSquared( pos, polyline[i-1] , polyline[i] - polyline[i-1] ) );
	}
	if( polyline.Cyclic() ) {
		dist = Min( dist, 
			PointLineSegmentDistanceSquared( pos, polyline[polyline.Size()-1] , polyline[0] - polyline[polyline.Size()-1] ) );
	}
	return dist;
}
	
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*POLYLINE_H*/
