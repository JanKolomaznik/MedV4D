#ifndef BSPLINE_H
#define BSPLINE_H

#include "Imaging/PointSet.h"
#include "Imaging/Polyline.h"
#include <cmath>
#include "common/Common.h"
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
				_THROW_ ErrorHandling::EBadIndex(); 
		}
	const ValueType &
	operator[]( unsigned idx ) const
		{ 
			if( idx <= Degree ) {
				return _values[ idx ]; 
			} else 
				_THROW_ ErrorHandling::EBadIndex(); 
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

			double scale = 1.0/6.0;

			values[ 0 ] = scale * ( - 3*v[1] +  6*v[0] - 3 );
			values[ 1 ] = scale * (   9*v[1] - 12*v[0]     );
			values[ 2 ] = scale * ( - 9*v[1] +  6*v[0] + 3 );
			values[ 3 ] = scale * (   3*v[1]               );

			return true;
		}

	template< typename ValueType >
	static bool
	ValuesAndDerivationsAtPoint( 
			double t, 
			BasisFunctionValues< ValueType, Degree > &values, 
			BasisFunctionValues< ValueType, Degree > &dvalues )
		{
			ValuesAtPoint( t, values );
			return DerivationAtPoint( t, dvalues );	
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
	typedef Polyline< CoordType, Dim >				SamplePointSet;
	typedef CoordType						Type;
	static const unsigned 						Degree = CurveBasis::Degree;
	static const unsigned 						Dimension	= Dim;		

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

	const PointSet< CoordType, Dim > &
	GetSampleDerivations()const
		{ return _sampleDerivationCache; }

	const SamplePointSet &
	GetSamplePoints()const
		{ return _samplePointCache; }

	unsigned
	GetLastSampleFrequency() const
		{ return _lastSampleFrequency; }

	const BFValVector &
	GetLastBasisFunctionValues() const
		{ return _lastBasisFunctionValues; }

	const BFValVector &
	GetLastBasisFunctionDerivationValues() const
		{ return _lastBasisFunctionDerivationValues; }

	void
	SetCyclic( bool cyclic = true )
		{ _cyclic = cyclic; _samplePointCache.SetCyclic( cyclic ); }

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
	
	void
	SampleWithFunctionValues( unsigned sampleFrequency, PointSet< CoordType, Dim > &points, const typename BSpline< CoordType, Dim >::BFValVector &values );

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


template < typename CurveType >
void
SplitSegment( CurveType &curve, int segment )
{
	int idx = segment + CurveType::CurveBasis::HalfDegree;
	if( curve.Cyclic() ) {
		typename CurveType::PointType newPoint = 0.5f * (curve[ MOD(idx, curve.Size()) ]+curve[ MOD(idx+1, curve.Size()) ]);
		curve.InsertPoint( MOD(idx+1, curve.Size()), newPoint );
		curve.ReSample();
	} else {
		_THROW_ ErrorHandling::ENotFinished( "SplitSegment() - Handling noncyclic splines" );
	}
}

template < typename CurveType >
void
JoinSegments( CurveType &curve, int segment )
{
	int idx = segment + (CurveType::Degree+1 + 1)/2;
	curve.RemovePoint( MOD(idx, curve.Size()) );
	curve.ReSample();
}

template< typename CoordType >
bool
CheckSegmentIntersection( BSpline< CoordType, 2 > &curve, int32 segment1, int32 segment2 )
{
	const typename BSpline< CoordType, 2 >::SamplePointSet &samples = curve.GetSamplePoints();
	unsigned frq = curve.GetLastSampleFrequency();
	if( segment1 > segment2 ) {
		int32 tmp = segment2;
		segment2 = segment1;
		segment1 = tmp;
	}

	if( segment1 == 0 && segment2 == (int32)curve.GetSegmentCount() - 1 ) {
		for( unsigned i = segment1 * frq; i < (segment1+1) * frq; ++i ) {
			for( unsigned j = Max(i+2, segment2 * frq); j < (segment2+1) * frq -1; ++j ) {
				if( LineIntersectionTest( samples.GetPointCyclic(i), samples.GetPointCyclic(i+1), samples.GetPointCyclic(j), samples.GetPointCyclic(j+1) ) ) {
					return true;
				}
			}
		}
		return false;
	}

	for( unsigned i = segment1 * frq; i < (segment1+1) * frq; ++i ) {
		for( unsigned j = Max(i+2, segment2 * frq); j < (segment2+1) * frq; ++j ) {
			if( LineIntersectionTest( samples.GetPointCyclic(i), samples.GetPointCyclic(i+1), samples.GetPointCyclic(j), samples.GetPointCyclic(j+1) ) ) {
				return true;
			}
		}
	}
	return false;
}

template< typename CoordType >
bool
FindBSplineSelfIntersection( BSpline< CoordType, 2 > &curve, CoordInt2D &segIndices )
{
	segIndices = CoordInt2D( -1, -1 );
	for( unsigned i = 0; i < curve.GetSegmentCount(); ++i ) {
		for( unsigned j = i; j < curve.GetSegmentCount(); ++j ) {
			if( CheckSegmentIntersection( curve, i, j ) ) {
				segIndices[0] = i;
				segIndices[1] = j;
				return true;
			}
		}
	}

	return false;
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
float32
BSplineLength( BSpline< CoordType, Dim > &curve )
{
	float32 length = 0;
	for( unsigned i = 0; i < curve.GetSegmentCount(); ++i ) {
		length += BSplineSegmentLength( curve, i );
	}
	return length;
}

template< typename CoordType, unsigned Dim >
void
FindBSplineSegmentLengthExtremes( BSpline< CoordType, Dim > &curve, unsigned &maxIdx, float32 &maxVal, unsigned &minIdx, float32 &minVal )
{
	/*float32	 first = BSplineSegmentLength( curve, 0 );
	float32	 len = first;
	maxIdx = 0;
	maxVal = len;
	minIdx = 0;
	minVal = first + BSplineSegmentLength( curve, 1 );
	float32 previous = first;
	for( unsigned i=1; i < curve.GetSegmentCount(); ++i ) {
		len = BSplineSegmentLength( curve, i );
		if( len > maxVal ) {
			maxVal = len;
			maxIdx = i;
		} 
		if( previous + len < minVal ) {
			minVal = previous + len;
			minIdx = i-1;
		}
		previous = len;
	}
	if( previous + first < minVal ) {
			minVal = previous + first;
			minIdx = curve.GetSegmentCount();
	}*/
	float32	 len =  BSplineSegmentLength( curve, 0 );
	maxIdx = 0;
	maxVal = len;
	minIdx = 0;
	minVal = len;
	for( unsigned i=1; i < curve.GetSegmentCount(); ++i ) {
		len = BSplineSegmentLength( curve, i );
		if( len > maxVal ) {
			maxVal = len;
			maxIdx = i;
		} 
		if( len < minVal ) {
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

template < typename CurveType >
void
PrintCurveSegmentPoints( std::ostream &stream, const CurveType &curve )
{
	const typename CurveType::SamplePointSet &points = curve.GetSamplePoints();
	unsigned frq = curve.GetLastSampleFrequency();
	PrintPointSet( stream, points, frq );

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
