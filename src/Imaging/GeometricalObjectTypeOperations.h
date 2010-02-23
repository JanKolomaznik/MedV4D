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
	NTID_MASK 			= 0xFF,
	//DIMENSION_MASK 			= 0xF0,
	//DIMENSION_SHIFT			= 4,

	GTID_AGEOMETRICAL_OBJECT 	= 1 << 8,
	GTID_POINT_SET			= 2 << 8,
	GTID_BSPLINE			= 3 << 8,



};


#define GEOMETRY_TYPE_SWITCH_DEFAULT_MACRO( SWITCH, DEFAULT, ... ) \
	switch( SWITCH ) {\
	default: DEFAULT;\
	}

#define GEOMETRY_TYPE_SWITCH_MACRO( SWITCH, ... ) \
	GEOMETRY_TYPE_SWITCH_DEFAULT_MACRO( SWITCH, ASSERT( false ), __VA_ARGS__ )


enum GeometryMagicNumbers
{
	GMN_BEGIN_ATRIBUTES		= 0xFFFFAAAA,
	GMN_END_ATRIBUTES		= 0x1111BBBB,
	GMN_BEGIN_DATA			= 0x1257ABE6,
	GMN_END_DATA			= 0x16516CD8
};

template< unsigned size >
struct DummySpace
{
	uint32	data[ size ];
};

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

