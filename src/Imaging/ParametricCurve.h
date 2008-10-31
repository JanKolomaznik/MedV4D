#ifndef PARAMETRIC_CURVE
#define PARAMETRIC_CURVE

#include "Imaging/PointSet.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ParametricCurve.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{


template < typename CoordType, unsigned Dim >
class ParametricCurve: public PointSet< CoordType, Dim >
{
public:

};

	
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*PARAMETRIC_CURVE*/
