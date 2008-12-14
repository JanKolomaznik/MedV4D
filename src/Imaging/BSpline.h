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
	static const int HalfDegree = Degree / 2;
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
	typedef std::vector< BFunctionValues >				BFValVector;
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
	ReSample();

	void
	SampleWithDerivations( unsigned frequency );

	void
	ReSampleWithDerivations();

	void
	ResetSamples();
	
	void
	ResetSamplesDerivations();

	void
	SplitSegment( int segment );

	void
	JoinSegment( int segment );

	const PointSet< CoordType, Dim > &
	GetSampleDerivations()const
		{ return _sampleDerivationCache; }

	const PointSet< CoordType, Dim > &
	GetSamplePoints()const
		{ return _samplePointCache; }

	unsigned
	GetLastSampleFrequency() const
		{ return _lastSampleFrequency; }

	const BFValVector &
	GetLastBasisFunctionValues() const
		{ return _lastBasisFunctionValues; }

	void
	SetCyclic( bool cyclic = true )
		{ _cyclic = cyclic; }

	bool
	Cyclic() const
		{ return _cyclic; }

	unsigned
	GetSegmentCount() const
		{  
			if( _cyclic ) {
				return GetNormalSegmentCount() + GetCyclicSegmentCount();
			} else {
				return GetNormalSegmentCount() + 2*GetEndSegmentCount();
			}
		}

	/**
	 * Returns number of uniform spline "normal" segments (no multiple points, acyclic).
	 **/
	unsigned
	GetNormalSegmentCount() const
		{ return Max( 0, static_cast<int>( this->Size() - CurveBasis::Degree ) ); }
	/**
	 * Returns number of uniform spline segments, which make it cyclical.
	 **/
	unsigned
	GetCyclicSegmentCount() const
		{ return CurveBasis::Degree; }
	/**
	 * Returns number of uniform spline segments produced by multipliing points on end.
	 **/
	unsigned
	GetEndSegmentCount() const
		{ return CurveBasis::Degree; }
protected:
	unsigned
	SampleUniformSpline( unsigned firstPoint, PointSet< CoordType, Dim > &points, const BFValVector &values );

	unsigned
	SampleUniformSplineCyclicEnd( unsigned firstPoint, PointSet< CoordType, Dim > &points, const BFValVector &values );

	unsigned
	SampleUniformSplineACyclicBegin( unsigned firstPoint, PointSet< CoordType, Dim > &points, const BFValVector &values );

	unsigned
	SampleUniformSplineACyclicEnd( unsigned firstPoint, PointSet< CoordType, Dim > &points, const BFValVector &values );

	inline PointType
	EvaluateCurve( int segment, const BFunctionValues &values );

	inline PointType
	EvaluateCyclicCurve( int segment, const BFunctionValues &values );

	inline PointType
	EvaluateACyclicCurve( int segment, const BFunctionValues &values );


	bool 			_cyclic;
	SamplePointSet		_samplePointCache;
	SamplePointSet		_sampleDerivationCache;

	BFValVector		_lastBasisFunctionValues;
	BFValVector		_lastBasisFunctionDerivationValues;

	unsigned 		_lastSampleFrequency;

};

template< typename CoordType >
CoordInt2D
FindBSplineSelfIntersection( BSpline< CoordType, 2 > &curve )
{
	const typename BSpline< CoordType, 2 >::SamplePointSet &samples = curve.GetSamplePoints();
	CoordInt2D result = CoordInt2D( -1, -1 );
	for( unsigned i = 0; i < samples.Size()-3; ++i ) {
		for( unsigned j = i+2; j < samples.Size()-1; ++j ) {
			if( LineIntersectionTest( samples[i], samples[i+1], 
						samples[j], samples[j+1] ) ) 
			{
				result[0] = i / curve.GetLastSampleFrequency();
				result[1] = j / curve.GetLastSampleFrequency();
				return result;
			}
		}
	}

	return result;
}

template< typename CoordType, unsigned Dim >
float32
BSplineSegmentLength( BSpline< CoordType, Dim > &curve, unsigned segment )
{
	const typename BSpline< CoordType, Dim >::SamplePointSet &samples = curve.GetSamplePoints();
	segment = segment % curve.GetSegmentCount();
	
	float32 length = 0;
	
	for( unsigned i = segment * curve.GetLastSampleFrequency(); i < (segment+1) * curve.GetLastSampleFrequency(); ++i ) {
		typename BSpline< CoordType, Dim >::PointType diff = samples[i] - samples[MOD( i+1, samples.Size() )];
		length += sqrt( diff * diff );
	}

	return length;
}

template< typename CoordType, unsigned Dim >
void
FindBSplineSegmentLengthExtremes( BSpline< CoordType, Dim > &curve, unsigned &maxIdx, float32 &maxVal, unsigned &minIdx, float32 &minVal )
{
	float32	 len = BSplineSegmentLength( curve, 0 );
	maxIdx = 0;
	maxVal = len;
	minIdx = 0;
	minVal = len;
	for( unsigned i=1; i < curve.GetSegmentCount(); ++i ) {
		len = BSplineSegmentLength( curve, i );
		if( len > maxVal ) {
			maxVal = len;
			maxIdx = i;
		} else if( len < minVal ) {
			minVal = len;
			minIdx = i;
		}
	}
}

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
