#ifndef POLYLINE_H
#define POLYLINE_H

#include "Imaging/PointSet.h"

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
	typedef Coordinates< CoordType, Dim > 	PointType;
	typedef CoordType			Type;
	static const unsigned Dimension	= Dim;		

};

	
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*POLYLINE_H*/
