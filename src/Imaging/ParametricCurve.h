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

	ParametricCurve();

	ParametricCurve( const PointSet< CoordType, Dim > & points );

	PointType
	PointByParameter( double t )const;

	bool
	DerivationAtPoint( double t, PointType &derivation )const;

	bool
	PointAndDerivationAtPoint( double t, PointType &, PointType &derivation )const;
	
	void
	Sample( unsigned frequency );

	void
	SampleWithDerivations( unsigned frequency );

	void
	ResetSamples();
protected:
	bool 				_cyclic;
	PointSet< CoordType, Dim >	_samplePointCache;
	PointSet< CoordType, Dim >	_sampleDerivationCache;

	CurveCore			_curveCore;
};


template < typename CoordType, unsigned Dim, typename CurveCore >
ParametricCurve< CoordType, Dim, CurveCore >
::ParametricCurve()
{

}

template < typename CoordType, unsigned Dim, typename CurveCore >
ParametricCurve< CoordType, Dim, CurveCore >
::ParametricCurve( const PointSet< CoordType, Dim > & points )
{

}

template < typename CoordType, unsigned Dim, typename CurveCore >
PointType
ParametricCurve< CoordType, Dim, CurveCore >
::PointByParameter( double t )const
{

}

template < typename CoordType, unsigned Dim, typename CurveCore >
bool
ParametricCurve< CoordType, Dim, CurveCore >
::DerivationAtPoint( double t, PointType &derivation )const
{

}

template < typename CoordType, unsigned Dim, typename CurveCore >
bool
ParametricCurve< CoordType, Dim, CurveCore >
::PointAndDerivationAtPoint( double t, PointType &, PointType &derivation )const
{

}

template < typename CoordType, unsigned Dim, typename CurveCore >
void
ParametricCurve< CoordType, Dim, CurveCore >
::Sample( unsigned frequency )
{

}

template < typename CoordType, unsigned Dim, typename CurveCore >
void
ParametricCurve< CoordType, Dim, CurveCore >
::SampleWithDerivations( unsigned frequency )
{

}

template < typename CoordType, unsigned Dim, typename CurveCore >
void
ParametricCurve< CoordType, Dim, CurveCore >
::ResetSamples()
{

}

	
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*PARAMETRIC_CURVE*/
