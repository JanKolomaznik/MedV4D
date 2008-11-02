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


template < typename CoordType, unsigned Dim, typename CurveCore >
class ParametricCurve: public PointSet< CoordType, Dim >
{
public:
	friend CurveCore;

	PointType
	PointByParameter( double t )const;

	bool
	DerivationAtPoint( double t, PointType &derivation )const;	
	
	Sample( unsigned frequency );

	SampleWithDerivations( unsigned frequency );

	ResetSamples();
protected:
	bool 				_cyclic;
	PointSet< CoordType, Dim >	_samplePointCache;
};

	
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*PARAMETRIC_CURVE*/
