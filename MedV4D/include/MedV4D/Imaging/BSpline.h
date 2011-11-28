#ifndef BSPLINE_H
#define BSPLINE_H

#include "MedV4D/Imaging/PointSet.h"
#include "MedV4D/Imaging/Polyline.h"
#include <cmath>
#include "MedV4D/Common/Common.h"
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

namespace Imaging {
namespace Geometry {


template< typename ValueType, unsigned Degree >
struct BasisFunctionValues {
        ValueType &
        operator[] ( unsigned idx ) {
                if ( idx <= Degree ) {
                        return _values[ idx ];
                } else
                        _THROW_ ErrorHandling::EBadIndex();
        }
        const ValueType &
        operator[] ( unsigned idx ) const {
                if ( idx <= Degree ) {
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
        ValuesAtPoint ( double t, BasisFunctionValues< ValueType, Degree > &values ) {
                double v[ Degree ];
                BSplineBasis::ParameterPowers ( t, v );

                double scale = 1.0/6.0;

                values[ 0 ] = static_cast<ValueType> ( scale * ( -   v[2] + 3*v[1] - 3*v[0] + 1 ) );
                values[ 1 ] = static_cast<ValueType> ( scale * ( 3*v[2] - 6*v[1]          + 4 ) );
                values[ 2 ] = static_cast<ValueType> ( scale * ( - 3*v[2] + 3*v[1] + 3*v[0] + 1 ) );
                values[ 3 ] = static_cast<ValueType> ( scale * ( v[2] ) );

        }

        template< typename ValueType >
        static bool
        DerivationsAtPoint ( double t, BasisFunctionValues< ValueType, Degree > &values ) {
                double v[ Degree ];
                BSplineBasis::ParameterPowers ( t, v );

                double scale = 1.0/6.0;

                values[ 0 ] = static_cast<float32> ( scale * ( - 3*v[1] +  6*v[0] - 3 ) );
                values[ 1 ] = static_cast<float32> ( scale * ( 9*v[1] - 12*v[0] ) );
                values[ 2 ] = static_cast<float32> ( scale * ( - 9*v[1] +  6*v[0] + 3 ) );
                values[ 3 ] = static_cast<float32> ( scale * ( 3*v[1] ) );

                return true;
        }

        template< typename ValueType >
        static bool
        ValuesAndDerivationsAtPoint (
                double t,
                BasisFunctionValues< ValueType, Degree > &values,
                BasisFunctionValues< ValueType, Degree > &dvalues ) {
                ValuesAtPoint ( t, values );
                return DerivationAtPoint ( t, dvalues );
        }

protected:
        static void
        ParameterPowers ( double t, double vector[] ) {
                vector[ 0 ] = t;
                for ( int i=1; i < Degree; ++i ) {
                        vector[ i ] = vector[ i-1 ] * t;
                }
        }

};

template < typename VectorType >
class BSpline: public PointSet< VectorType >
{
public:
        typedef typename VectorType::CoordinateType			Type;
        typedef BSplineBasis						CurveBasis;
        typedef BasisFunctionValues< Type, CurveBasis::Degree > 	BFunctionValues;
        typedef std::vector< BFunctionValues >				BFValVector;
        typedef PointSet< VectorType > 					Predecessor;
        typedef VectorType 						PointType;
        typedef Polyline< VectorType >					SamplePointSet;
        typedef BSpline< VectorType >					ThisType;
        static const unsigned 						Degree = CurveBasis::Degree;
        static const unsigned 						Dimension = VectorType::Dimension;

        friend void SerializeGeometryObject< VectorType > ( M4D::IO::OutStream &stream, const ThisType &obj );
        friend void DeserializeGeometryObject< VectorType > ( M4D::IO::InStream &stream, ThisType * &obj );

        BSpline();

        BSpline ( const PointSet< VectorType > & points );

        PointType
        PointByParameter ( double t ) const;

        bool
        DerivationAtPoint ( double t, PointType &derivation ) const;

        bool
        PointAndDerivationAtPoint ( double t, PointType &point, PointType &derivation ) const;

        void
        Sample ( unsigned frequency );

        void
        ReSample();

        void
        SampleWithDerivations ( unsigned frequency );

        void
        ReSampleWithDerivations();

        void
        ResetSamples();

        void
        ResetSamplesDerivations();

        const PointSet< VectorType > &
        GetSampleDerivations() const {
                return _sampleDerivationCache;
        }

        const SamplePointSet &
        GetSamplePoints() const {
                return _samplePointCache;
        }

        unsigned
        GetLastSampleFrequency() const {
                return _lastSampleFrequency;
        }

        const BFValVector &
        GetLastBasisFunctionValues() const {
                ASSERT ( _lastBasisFunctionValues.size() == _lastSampleFrequency );
                return _lastBasisFunctionValues;
        }

        const BFValVector &
        GetLastBasisFunctionDerivationValues() const {
                ASSERT ( _lastBasisFunctionDerivationValues.size() == _lastSampleFrequency );
                return _lastBasisFunctionDerivationValues;
        }

        void
        SetCyclic ( bool cyclic = true ) {
                _cyclic = cyclic;
                _samplePointCache.SetCyclic ( cyclic );
        }

        bool
        Cyclic() const {
                return _cyclic;
        }

        unsigned
        GetSegmentCount() const {
                if ( _cyclic ) {
                        return GetNormalSegmentCount() + GetCyclicSegmentCount();
                } else {
                        return GetNormalSegmentCount() + 2*GetEndSegmentCount();
                }
        }

        /**
         * Returns number of uniform spline "normal" segments (no multiple points, acyclic).
         **/
        unsigned
        GetNormalSegmentCount() const {
                return max ( 0, static_cast<int> ( this->Size() - CurveBasis::Degree ) );
        }
        /**
         * Returns number of uniform spline segments, which make it cyclical.
         **/
        unsigned
        GetCyclicSegmentCount() const {
                return CurveBasis::Degree;
        }
        /**
         * Returns number of uniform spline segments produced by multipliing points on end.
         **/
        unsigned
        GetEndSegmentCount() const {
                return CurveBasis::Degree;
        }
protected:

        void
        SampleWithFunctionValues ( unsigned sampleFrequency, PointSet< VectorType > &points, const typename BSpline< VectorType >::BFValVector &values );

        unsigned
        SampleUniformSpline ( unsigned firstPoint, PointSet< VectorType > &points, const BFValVector &values );

        unsigned
        SampleUniformSplineCyclicEnd ( unsigned firstPoint, PointSet< VectorType > &points, const BFValVector &values );

        unsigned
        SampleUniformSplineACyclicBegin ( unsigned firstPoint, PointSet< VectorType > &points, const BFValVector &values );

        unsigned
        SampleUniformSplineACyclicEnd ( unsigned firstPoint, PointSet< VectorType > &points, const BFValVector &values );

        inline PointType
        EvaluateCurve ( int segment, const BFunctionValues &values );

        inline PointType
        EvaluateCyclicCurve ( int segment, const BFunctionValues &values );

        inline PointType
        EvaluateACyclicCurve ( int segment, const BFunctionValues &values );


        bool 			_cyclic;
        SamplePointSet		_samplePointCache;
        SamplePointSet		_sampleDerivationCache;

        BFValVector		_lastBasisFunctionValues;
        BFValVector		_lastBasisFunctionDerivationValues;

        uint32	 		_lastSampleFrequency;

};


template < typename CurveType >
void
SplitSegment ( CurveType &curve, int segment )
{
        int idx = segment + CurveType::CurveBasis::HalfDegree;
        if ( curve.Cyclic() ) {
                typename CurveType::PointType newPoint = 0.5f * ( curve[ MOD ( idx, curve.Size() ) ]+curve[ MOD ( idx+1, curve.Size() ) ] );
                curve.InsertPoint ( MOD ( idx+1, curve.Size() ), newPoint );
                curve.ReSample();
        } else {
                _THROW_ ErrorHandling::ENotFinished ( "SplitSegment() - Handling noncyclic splines" );
        }
}

template < typename CurveType >
void
JoinSegments ( CurveType &curve, int segment )
{
        int idx = segment + ( CurveType::Degree+1 + 1 ) /2;
        curve.RemovePoint ( MOD ( idx, curve.Size() ) );
        curve.ReSample();
}

template< typename CoordType >
bool
CheckSegmentIntersection ( BSpline< Vector< CoordType, 2 > > &curve, int32 segment1, int32 segment2 )
{
        const typename BSpline< Vector< CoordType, 2 > >::SamplePointSet &samples = curve.GetSamplePoints();
        unsigned frq = curve.GetLastSampleFrequency();
        if ( segment1 > segment2 ) {
                int32 tmp = segment2;
                segment2 = segment1;
                segment1 = tmp;
        }

        if ( segment1 == 0 && segment2 == ( int32 ) curve.GetSegmentCount() - 1 ) {
                for ( unsigned i = segment1 * frq; i < ( segment1+1 ) * frq; ++i ) {
                        for ( unsigned j = max ( i+2, segment2 * frq ); j < ( segment2+1 ) * frq -1; ++j ) {
                                if ( LineIntersectionTest ( samples.GetPointCyclic ( i ), samples.GetPointCyclic ( i+1 ), samples.GetPointCyclic ( j ), samples.GetPointCyclic ( j+1 ) ) ) {
                                        return true;
                                }
                        }
                }
                return false;
        }

        for ( unsigned i = segment1 * frq; i < ( segment1+1 ) * frq; ++i ) {
                for ( unsigned j = max ( i+2, segment2 * frq ); j < ( segment2+1 ) * frq; ++j ) {
                        if ( LineIntersectionTest ( samples.GetPointCyclic ( i ), samples.GetPointCyclic ( i+1 ), samples.GetPointCyclic ( j ), samples.GetPointCyclic ( j+1 ) ) ) {
                                return true;
                        }
                }
        }
        return false;
}

template< typename CoordType >
bool
FindBSplineSelfIntersection ( BSpline< Vector< CoordType, 2 > > &curve, CoordInt2D &segIndices )
{
        segIndices = CoordInt2D ( -1, -1 );
        for ( unsigned i = 0; i < curve.GetSegmentCount(); ++i ) {
                for ( unsigned j = i; j < curve.GetSegmentCount(); ++j ) {
                        if ( CheckSegmentIntersection ( curve, i, j ) ) {
                                segIndices[0] = i;
                                segIndices[1] = j;
                                return true;
                        }
                }
        }

        return false;
}

template< typename VectorType >
float32
BSplineSegmentLength ( BSpline< VectorType > &curve, unsigned segment )
{
        const typename BSpline< VectorType >::SamplePointSet &samples = curve.GetSamplePoints();
        segment = segment % curve.GetSegmentCount();

        float32 length = 0;

        for ( unsigned i = segment * curve.GetLastSampleFrequency(); i < ( segment+1 ) * curve.GetLastSampleFrequency(); ++i ) {
                typename BSpline< VectorType >::PointType diff = samples[i] - samples[MOD ( i+1, samples.Size() ) ];
                length += sqrt ( diff * diff );
        }

        return length;
}

template< typename VectorType >
float32
BSplineLength ( BSpline< VectorType > &curve )
{
        float32 length = 0;
        for ( unsigned i = 0; i < curve.GetSegmentCount(); ++i ) {
                length += BSplineSegmentLength ( curve, i );
        }
        return length;
}

template< typename VectorType >
void
FindBSplineSegmentLengthExtremes ( BSpline< VectorType > &curve, unsigned &maxIdx, float32 &maxVal, unsigned &minIdx, float32 &minVal )
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
        float32	 len =  BSplineSegmentLength ( curve, 0 );
        maxIdx = 0;
        maxVal = len;
        minIdx = 0;
        minVal = len;
        for ( unsigned i=1; i < curve.GetSegmentCount(); ++i ) {
                len = BSplineSegmentLength ( curve, i );
                if ( len > maxVal ) {
                        maxVal = len;
                        maxIdx = i;
                }
                if ( len < minVal ) {
                        minVal = len;
                        minIdx = i;
                }
        }
}

template < typename CurveType >
void
PrintCurve ( std::ostream &stream, const CurveType &curve )
{
        const typename CurveType::SamplePointSet &points = curve.GetSamplePoints();
        PrintPointSet ( stream, points );

        if ( curve.Cyclic() && ( points.Size() > 0 ) ) {
                stream << points[0] << std::endl;
        }
}

template < typename CurveType >
void
PrintCurveSegmentPoints ( std::ostream &stream, const CurveType &curve )
{
        const typename CurveType::SamplePointSet &points = curve.GetSamplePoints();
        unsigned frq = curve.GetLastSampleFrequency();
        PrintPointSet ( stream, points, frq );

        if ( curve.Cyclic() && ( points.Size() > 0 ) ) {
                stream << points[0] << std::endl;
        }
}

template< typename VectorType >
void
SerializeGeometryObject ( M4D::IO::OutStream &stream, const BSpline< VectorType > &obj )
{
        stream.Put<uint32> ( GMN_BEGIN_ATRIBUTES );
        stream.Put ( DummySpace< 5 >() );
        stream.Put<uint32> ( obj._pointCount );
        stream.Put<bool> ( obj._cyclic );
        stream.Put<uint32> ( obj._lastSampleFrequency );
        stream.Put<uint32> ( GMN_END_ATRIBUTES );

        stream.Put<uint32> ( GMN_BEGIN_DATA );
        for ( uint32 i = 0; i < obj._pointCount; ++i ) {
                stream.Put< VectorType > ( obj._points[i] );
        }
        stream.Put<uint32> ( GMN_END_DATA );
}

template< typename VectorType >
void
DeserializeGeometryObject ( M4D::IO::InStream &stream, BSpline< VectorType > * &obj )
{
        _THROW_ M4D::ErrorHandling::ETODO();
}

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

//include implementation
#include "MedV4D/Imaging/BSpline.tcc"

#endif /*BSPLINE_H*/
