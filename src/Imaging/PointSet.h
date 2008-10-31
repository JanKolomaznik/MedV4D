#ifndef POINT_SET_H
#define POINT_SET_H

#include "Imaging/GeometricalObject.h"
#include "Coordinates.h"

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

};

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*POINT_SET_H*/
