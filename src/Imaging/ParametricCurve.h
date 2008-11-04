#ifndef PARAMETRIC_CURVE
#define PARAMETRIC_CURVE

#include "Imaging/PointSet.h"
#include <cmath>

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

template< typename ValueType, unsigned Degree >
struct BasisFunctionValues
{
	ValueType &
	operator[]( unsigned idx ) 
		{ 
			if( idx <= Degree ) {
				return _values[ idx ]; 
			} else 
				throw ErrorHandling::EWrongIndex(); 
		}
	const ValueType &
	operator[]( unsigned idx ) const
		{ 
			if( idx <= Degree ) {
				return _values[ idx ]; 
			} else 
				throw ErrorHandling::EWrongIndex(); 
		}

private:
	ValueType	_values[ Degree + 1 ];
};

class BSplineBasis
{
public:
	static const unsigned SplineDegree = 3;

	template< typename ValueType >
	static void
	ValuesAtPoint( double t, BasisFunctionValues< ValueType, SplineDegree > &values )
		{

		}

	template< typename ValueType >
	static bool
	DerivationsAtPoint( double t, BasisFunctionValues< ValueType, SplineDegree > &values )
		{
			return true;
		}

	template< typename ValueType >
	static bool
	ValuesAndDerivationsAtPoint( 
			double t, 
			BasisFunctionValues< ValueType, SplineDegree > &values, 
			BasisFunctionValues< ValueType, SplineDegree > &dvalues )
		{
			return true;
		}
	

};

template < typename CoordType, unsigned Dim, typename CurveBasis >
class ParametricCurve: public PointSet< CoordType, Dim >
{
public:
	typedef BasisFunctionValues< CoordType, CurveBasis::SplineDegree > BFunctionValues;

	ParametricCurve();

	ParametricCurve( const PointSet< CoordType, Dim > & points );

	PointType
	PointByParameter( double t )const;

	bool
	DerivationAtPoint( double t, PointType &derivation )const;

	bool
	PointAndDerivationAtPoint( double t, PointType &point, PointType &derivation )const;
	
	void
	Sample( unsigned frequency );

	void
	SampleWithDerivations( unsigned frequency );

	void
	ResetSamples();
	
	void
	ResetSamplesDerivations();

	void
	SplitSegment( int segment );
protected:
	PointType
	EvaluateCurve( int segment, const BFunctionValues &values )
		{ 
			PointType result;
			for( unsigned i = 0; i <= CurveBasis::Degree; ++i ) {
				//TODO check cyclic
				result += values[i]*(this->_points[ (segment % _pointCount) + i ]);
			}
		}


	bool 				_cyclic;
	PointSet< CoordType, Dim >	_samplePointCache;
	PointSet< CoordType, Dim >	_sampleDerivationCache;

};


template < typename CoordType, unsigned Dim, typename CurveBasis >
ParametricCurve< CoordType, Dim, CurveBasis >
::ParametricCurve()
{

}

template < typename CoordType, unsigned Dim, typename CurveBasis >
ParametricCurve< CoordType, Dim, CurveBasis >
::ParametricCurve( const PointSet< CoordType, Dim > & points )
{

}

template < typename CoordType, unsigned Dim, typename CurveBasis >
PointType
ParametricCurve< CoordType, Dim, CurveBasis >
::PointByParameter( double t )const
{
	int segment_nmbr = floor( t );
	double relative_t = t - segment_nmbr;

	BFunctionValues values;

	CurveBasis::ValuesAtPoint( rt, values );

	return EvaluateCurve( segment_nmbr, values );
}

template < typename CoordType, unsigned Dim, typename CurveBasis >
bool
ParametricCurve< CoordType, Dim, CurveBasis >
::DerivationAtPoint( double t, PointType &derivation )const
{
	int segment_nmbr = floor( t );
	double relative_t = t - segment_nmbr;

	BFunctionValues dvalues;

	if ( CurveBasis::DerivationsAtPoint( rt, dvalues ) ) {
		derivation = EvaluateCurve( segment_nmbr, values );
		return true;
	} 

	return false;
}

template < typename CoordType, unsigned Dim, typename CurveBasis >
bool
ParametricCurve< CoordType, Dim, CurveBasis >
::PointAndDerivationAtPoint( double t, PointType &point, PointType &derivation )const
{
	int segment_nmbr = floor( t );
	double relative_t = t - segment_nmbr;

	BFunctionValues values;
	BFunctionValues dvalues;

	if ( CurveBasis::ValuesAndDerivationsAtPoint( rt, values, dvalues ) ) {
		point = EvaluateCurve( segment_nmbr, values );
		derivation = EvaluateCurve( segment_nmbr, dvalues );
		return true;
	} 

	point = EvaluateCurve( segment_nmbr, values );
	return false;
}

template < typename CoordType, unsigned Dim, typename CurveBasis >
void
ParametricCurve< CoordType, Dim, CurveBasis >
::Sample( unsigned frequency )
{
	//TODO check cyclic
	int32 sampleCount = frequency * (_pointCount - 1);
	_samplePointCache.Reserve( sampleCount );

	std::vector< BFunctionValues > precomputedFValues;
	precomputedFValues.reserve( frequency );

	double dt = 1.0 / frequency;
	double t = 0.0;
	for( unsigned i=0; i < frequency; ++i, t += dt ) {
		CurveBasis::ValuesAtPoint( t, precomputedFValues[ i ] );	
	}
	
	unsigned actualSample = 0;
	for( unsigned i = 0; i < _pointCount-1; ++i ) {
		for( unsigned j = 0; j < frequency; ++j, ++actualSample ) {
			_samplePointCache[ actualSample ] = EvaluateCurve( i, precomputedFValues[ j ] );
		}
	}

}

template < typename CoordType, unsigned Dim, typename CurveBasis >
void
ParametricCurve< CoordType, Dim, CurveBasis >
::SampleWithDerivations( unsigned frequency )
{

}

template < typename CoordType, unsigned Dim, typename CurveBasis >
void
ParametricCurve< CoordType, Dim, CurveBasis >
::ResetSamples()
{

}

template < typename CoordType, unsigned Dim, typename CurveBasis >
void
ParametricCurve< CoordType, Dim, CurveBasis >
::ResetSamplesDerivations()
{

}

template < typename CoordType, unsigned Dim, typename CurveBasis >
void
ParametricCurve< CoordType, Dim, CurveBasis >
::SplitSegment( int segment )
{

}


}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*PARAMETRIC_CURVE*/
