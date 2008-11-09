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

template < typename Type, typename TypeB >
Type
min( Type a,  TypeB b )
{ return (Type) (a<b ? a : b); }

template < typename Type, typename TypeB >
Type
max( Type a,  TypeB b )
{ return (Type) (a<b ? b : a); }

//**************************************************

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
	static const unsigned Degree = 3;

	template< typename ValueType >
	static void
	ValuesAtPoint( double t, BasisFunctionValues< ValueType, Degree > &values )
		{
			double v[ Degree ];
			BSplineBasis::ParameterPowers( t, v );
			
			values[ 0 ] = -   v[2] + 3*v[1] - 3*v[0] + 1;
			values[ 1 ] =   3*v[2] - 6*v[1]          + 4;
			values[ 2 ] = - 3*v[2] + 3*v[1] + 3*v[0] + 1;
			values[ 3 ] =     v[2]                      ;

		}

	template< typename ValueType >
	static bool
	DerivationsAtPoint( double t, BasisFunctionValues< ValueType, Degree > &values )
		{
			double v[ Degree ];
			BSplineBasis::ParameterPowers( t, v );
			

			return true;
		}

	template< typename ValueType >
	static bool
	ValuesAndDerivationsAtPoint( 
			double t, 
			BasisFunctionValues< ValueType, Degree > &values, 
			BasisFunctionValues< ValueType, Degree > &dvalues )
		{
			double v[ Degree ];
			BSplineBasis::ParameterPowers( t, v );
			

			return true;
		}
	
protected:
	static void
	ParameterPowers( double t, double vector[] )
		{ 
			vector[ 0 ] = t;
			for( unsigned i=1; i < Degree; ++i ) {
				vector[ i ] = vector[ i-1 ] * t;
			}
		}

};

template < typename CoordType, unsigned Dim, typename CurveBasis >
class ParametricCurve: public PointSet< CoordType, Dim >
{
public:
	typedef BasisFunctionValues< CoordType, CurveBasis::Degree > 	BFunctionValues;
	typedef PointSet< CoordType, Dim > 					Predecessor;
	typedef typename Predecessor::PointType PointType;

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

	const PointSet< CoordType, Dim > &
	GetSampleDerivations()const
		{ return _sampleDerivationCache; }

	const PointSet< CoordType, Dim > &
	GetSamplePoints()const
		{ return _samplePointCache; }
protected:
	static const unsigned curveEndSegCount = (CurveBasis::Degree - 1)/2;

	inline PointType
	EvaluateCurve( int segment, const BFunctionValues &values );

	inline PointType
	EvaluateCyclicCurve( int segment, const BFunctionValues &values );

	inline PointType
	EvaluateACyclicCurve( int segment, const BFunctionValues &values );

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
typename ParametricCurve< CoordType, Dim, CurveBasis >::PointType
ParametricCurve< CoordType, Dim, CurveBasis >
::PointByParameter( double t )const
{
	int segment_nmbr = floor( t );
	double relative_t = t - segment_nmbr;

	BFunctionValues values;

	CurveBasis::ValuesAtPoint( relative_t, values );

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

	if ( CurveBasis::DerivationsAtPoint( relative_t, dvalues ) ) {
		derivation = EvaluateCurve( segment_nmbr, dvalues );
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

	if ( CurveBasis::ValuesAndDerivationsAtPoint( relative_t, values, dvalues ) ) {
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
	int32 sampleCount = frequency * (this->_pointCount - 1);
	_samplePointCache.Reserve( sampleCount );

	std::vector< BFunctionValues > precomputedFValues;
	precomputedFValues.reserve( frequency );

	double dt = 1.0 / frequency;
	double t = 0.0;
	for( unsigned i=0; i < frequency; ++i, t += dt ) {
		CurveBasis::ValuesAtPoint( t, precomputedFValues[ i ] );	
	}
	
	unsigned actualSample = 0;
	for( unsigned i = 0; i < this->_pointCount-1; ++i ) {
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

template < typename CoordType, unsigned Dim, typename CurveBasis >
typename ParametricCurve< CoordType, Dim, CurveBasis >::PointType
ParametricCurve< CoordType, Dim, CurveBasis >
::EvaluateCurve( int segment, const BFunctionValues &values )
{
	if( _cyclic ) {
		return EvaluateCyclicCurve( segment, values );
	} else {
		return EvaluateACyclicCurve( segment, values );
	}
}

template < typename CoordType, unsigned Dim, typename CurveBasis >
typename ParametricCurve< CoordType, Dim, CurveBasis >::PointType
ParametricCurve< CoordType, Dim, CurveBasis >
::EvaluateCyclicCurve( int segment, const BFunctionValues &values )
{ 
	PointType result;

	if( segment < curveEndSegCount ) {
		for( unsigned i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ ((segment-curveEndSegCount) + i)%this->_pointCount ]);
		}
		return result;
	} 
	if( segment >= (this->_pointCount-curveEndSegCount) ) {
		for( unsigned i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ ((segment-curveEndSegCount) + i)%this->_pointCount ]);
		}
		return result;
	} 

	for( unsigned i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ (segment-curveEndSegCount) + i ]);
		}
	return result;
}

template < typename CoordType, unsigned Dim, typename CurveBasis >
typename ParametricCurve< CoordType, Dim, CurveBasis >::PointType
ParametricCurve< CoordType, Dim, CurveBasis >
::EvaluateACyclicCurve( int segment, const BFunctionValues &values )
{ 
	PointType result;

	if( segment < curveEndSegCount ) {
		for( unsigned i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ max(segment-curveEndSegCount+i,0) ]);
		}
		return result;
	} 
	if( segment >= (this->_pointCount-curveEndSegCount) ) {
		for( unsigned i = 0; i <= CurveBasis::Degree; ++i ) {
			//TODO check cyclic
			result += values[i]*(this->_points[ min(segment-curveEndSegCount+i,0) ]);
		}
		return result;
	} 

	for( unsigned i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ (segment-curveEndSegCount) + i ]);
		}
	return result;
}


}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*PARAMETRIC_CURVE*/
