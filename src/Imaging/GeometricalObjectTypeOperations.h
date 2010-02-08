#ifndef GEOMETRICAL_OBJECT_TYPE_OPERATIONS_H
#define GEOMETRICAL_OBJECT_TYPE_OPERATIONS_H

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file GeometricalObject.h 
 * @{ 
 **/

#include "common/Vector.h"

namespace Imaging
{
namespace Geometry
{

enum GeometryTypeID
{
	GTID_UNKNOWN			= 0x00,
	NTID_MASK 			= 0x0F,
	DIMENSION_MASK 			= 0x70,

	GTID_AGEOMETRICAL_OBJECT 	= 0x01,



};


#define GEOMETRY_TYPE_SWITCH_DEFAULT_MACRO( SWITCH, DEFAULT, ... ) \
	switch( SWITCH ) {\
	default: DEFAULT;\
	}

#define GEOMETRY_TYPE_SWITCH_MACRO( SWITCH, ... ) \
	GEOMETRY_TYPE_SWITCH_DEFAULT_MACRO( SWITCH, ASSERT( false ), __VA_ARGS__ )

/*template< typename GeometryType >
GeometryTypeID
GetGeometryObjectTypeID();*/
/*{
	return GTID_UNKNOWN;
}*/

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*GEOMETRICAL_OBJECT_TYPE_OPERATIONS_H*/

