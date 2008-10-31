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

namespace Imaging
{
namespace Geometry
{

class GeometricalObject
{
public:
	virtual ~GeometricalObject(){}
};

template< unsigned Dim >
class GeometricalObjectDim: public GeometricalObject
{
public:
	static const unsigned Dimension = Dim;
};


}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*GEOMETRICAL_OBJECT_H*/
