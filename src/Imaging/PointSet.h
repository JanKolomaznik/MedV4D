#ifndef POINT_SET_H
#define POINT_SET_H

#include "Imaging/GeometricalObject.h"
#include "Coordinates.h"
#include <vector>


namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file PointSet.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{

	
template < typename CoordType, unsigned Dim >
class PointSet: public GeometricalObjectDim< Dim >
{
public:
	typedef Coordinates< CoordType, Dim > PointType;

	void
	Reserve( unsigned size )
		{ _points.reserve( _pointCount = size ); }

	PointType &
	operator[]( unsigned idx ) 
		{ 
			if( idx < _pointCount ) {
				return _points[ idx ]; 
			} else 
				throw ErrorHandling::EWrongIndex(); 
		}

	const PointType &
	operator[]( unsigned idx ) const
		{ 
			if( idx < _pointCount ) {
				return _points[ idx ]; 
			} else 
				throw ErrorHandling::EWrongIndex(); 
		}
protected:
	std::vector< PointType >	_points;
	uint32				_pointCount;
};

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*POINT_SET_H*/
