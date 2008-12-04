#ifndef BSPLINE_H
#define BSPLINE_H

#include "Imaging/PointSet.h"
#include <cmath>
#include "Common.h"
#include <ostream>
#include <iomanip>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file BSpline.h 
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
	static const int Degree = 3;
	/**
	 * With how many segments has incident interval - in one direction
	 **/
	static const int SupportRadius = 1;

	template< typename ValueType >
	static void
	ValuesAtPoint( double t, BasisFunctionValues< ValueType, Degree > &values )
		{
			double v[ Degree ];
			BSplineBasis::ParameterPowers( t, v );
			
			double scale = 1.0/6.0;

			values[ 0 ] = scale * ( -   v[2] + 3*v[1] - 3*v[0] + 1 );
			values[ 1 ] = scale * (   3*v[2] - 6*v[1]          + 4 );
			values[ 2 ] = scale * ( - 3*v[2] + 3*v[1] + 3*v[0] + 1 );
			values[ 3 ] = scale * (     v[2]                       );

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
			for( int i=1; i < Degree; ++i ) {
				vector[ i ] = vector[ i-1 ] * t;
			}
		}

};

template < typename CoordType, unsigned Dim >
class BSpline: public PointSet< CoordType, Dim >
{
public:
	typedef BSplineBasis						CurveBasis;
	typedef BasisFunctionValues< CoordType, CurveBasis::Degree > 	BFunctionValues;
	typedef PointSet< CoordType, Dim > 				Predecessor;
	typedef typename Predecessor::PointType 			PointType;
	typedef PointSet< CoordType, Dim >				SamplePointSet;
	typedef CoordType						Type;
	static const unsigned Dimension	= Dim;		

	BSpline();

	BSpline( const PointSet< CoordType, Dim > & points );

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

	void
	SetCyclic( bool cyclic = true )
		{ _cyclic = cyclic; }

	bool
	Cyclic() const
		{ return _cyclic; }
protected:
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

template < typename CurveType >
void
PrintCurve( std::ostream &stream, const CurveType &curve )
{
	const typename CurveType::SamplePointSet &points = curve.GetSamplePoints();
	PrintPointSet( stream, points );

	if( curve.Cyclic() && (points.Size() > 0) ) {
		stream << points[0] << std::endl;
	}
}

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

//include implementation
#include "Imaging/BSpline.tcc"

#endif /*BSPLINE_H*/
