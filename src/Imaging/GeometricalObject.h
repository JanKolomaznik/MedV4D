#ifndef GEOMETRICAL_OBJECT_H
#define GEOMETRICAL_OBJECT_H

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file GeometricalObject.h 
 * @{ 
 **/

#include "Vector.h"

namespace Imaging
{
namespace Geometry
{

class AGeometricalObject
{
public:
	virtual ~AGeometricalObject(){}
};

template< unsigned Dim >
class AGeometricalObjectDim: public AGeometricalObject
{
public:
	static const unsigned Dimension = Dim;
};

template< typename CoordType, unsigned Dim >
class AGeometricalObjectDimPrec: public AGeometricalObjectDim< Dim >
{
public:
	typedef CoordType			Type;
	typedef Vector< Type, Dim > 	PointType;
	
};

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*GEOMETRICAL_OBJECT_H*/
