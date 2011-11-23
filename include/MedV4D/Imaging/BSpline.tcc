#ifndef BSPLINE_H
#error File BSpline.tcc cannot be included directly!
#else

#include <cmath>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ParametricCurve.tcc
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{


template < typename VectorType >
BSpline< VectorType >
::BSpline(): _cyclic( false ), _lastSampleFrequency( 2 )
{
	_samplePointCache.SetCyclic( _cyclic );
}

template < typename VectorType >
BSpline< VectorType >
::BSpline( const PointSet< VectorType > & points ): _cyclic( false ), _lastSampleFrequency( 2 )
{
	_samplePointCache.SetCyclic( _cyclic );
}

template < typename VectorType >
typename BSpline< VectorType >::PointType
BSpline< VectorType >
::PointByParameter( double t )const
{
	int segment_nmbr = M4D::floor( t );
	double relative_t = t - segment_nmbr;

	BFunctionValues values;

	CurveBasis::ValuesAtPoint( relative_t, values );

	return EvaluateCurve( segment_nmbr, values );
}

template < typename VectorType >
bool
BSpline< VectorType >
::DerivationAtPoint( double t, PointType &derivation )const
{
	int segment_nmbr = M4D::floor( t );
	double relative_t = t - segment_nmbr;

	BFunctionValues dvalues;

	if ( CurveBasis::DerivationsAtPoint( relative_t, dvalues ) ) {
		derivation = EvaluateCurve( segment_nmbr, dvalues );
		return true;
	} 

	return false;
}

template < typename VectorType >
bool
BSpline< VectorType >
::PointAndDerivationAtPoint( double t, PointType &point, PointType &derivation )const
{
	int segment_nmbr = M4D::floor( t );
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

template < typename VectorType >
void
BSpline< VectorType >
::Sample( unsigned frequency )
{
	//TODO check
	_lastSampleFrequency = frequency;
	ReSample();
}

template < typename VectorType >
void
BSpline< VectorType >
::ReSample()
{
	if( this->_pointCount <= 1 ) {
		return;
	}
	
	//Precompute basis functions values
	_lastBasisFunctionValues.resize( _lastSampleFrequency );
	double dt = 1.0 / _lastSampleFrequency;
	double t = 0.0;
	for( unsigned i=0; i < _lastSampleFrequency; ++i, t += dt ) {
		CurveBasis::ValuesAtPoint( t, _lastBasisFunctionValues[ i ] );	
	}

	SampleWithFunctionValues( _lastSampleFrequency, _samplePointCache, _lastBasisFunctionValues );
	/*if( _cyclic ) {
		int32 sampleCount = GetSegmentCount() * _lastSampleFrequency;
		_samplePointCache.Resize( sampleCount );

		unsigned last =	SampleUniformSpline( 0, _samplePointCache, _lastBasisFunctionValues );
		SampleUniformSplineCyclicEnd( last, _samplePointCache, _lastBasisFunctionValues );
	} else {
		int32 sampleCount = GetSegmentCount() * _lastSampleFrequency + 1;
		_samplePointCache.Resize( sampleCount );

		unsigned last =	SampleUniformSplineACyclicBegin( 0, _samplePointCache, _lastBasisFunctionValues );
		last = SampleUniformSpline( last, _samplePointCache, _lastBasisFunctionValues );
		SampleUniformSplineACyclicEnd( last, _samplePointCache, _lastBasisFunctionValues );
	}*/
}

template < typename VectorType >
void
BSpline< VectorType >
::SampleWithFunctionValues( unsigned sampleFrequency, PointSet< VectorType > &points, const typename BSpline< VectorType >::BFValVector &values )
{
	if( _cyclic ) {
		int32 sampleCount = GetSegmentCount() * sampleFrequency;
		points.Resize( sampleCount );

		unsigned last =	SampleUniformSpline( 0, points, values );
		SampleUniformSplineCyclicEnd( last, points, values );
	} else {
		int32 sampleCount = GetSegmentCount() * sampleFrequency + 1;
		points.Resize( sampleCount );

		unsigned last =	SampleUniformSplineACyclicBegin( 0, points, values );
		last = SampleUniformSpline( last, points, values );
		SampleUniformSplineACyclicEnd( last, points, values );
	}

}

template < typename VectorType >
unsigned
BSpline< VectorType >
::SampleUniformSpline( unsigned firstPoint, PointSet< VectorType > &points, const typename BSpline< VectorType >::BFValVector &values )
{
	unsigned actualSample = firstPoint;
	for( int i = 0; i < (int)(this->Size()-CurveBasis::Degree); ++i ) {
		for( unsigned j = 0; j < _lastSampleFrequency; ++j, ++actualSample ) {
			points[ actualSample ] = EvaluateCurve( i, values[ j ] );
		}
	}
	return actualSample;
}

template < typename VectorType >
unsigned
BSpline< VectorType >
::SampleUniformSplineCyclicEnd( unsigned firstPoint, PointSet< VectorType > &points, const typename BSpline< VectorType >::BFValVector &values )
{
	unsigned actualSample = firstPoint;
	for( int i = 0; i < CurveBasis::Degree; ++i ) {
		for( unsigned j = 0; j < _lastSampleFrequency; ++j, ++actualSample ) {
			points[ actualSample ] = EvaluateCyclicCurve( i+this->Size()-CurveBasis::Degree, values[ j ] );
		}
	}
	return actualSample;
}

template < typename VectorType >
unsigned
BSpline< VectorType >
::SampleUniformSplineACyclicBegin( unsigned firstPoint, PointSet< VectorType > &points, const typename BSpline< VectorType >::BFValVector &values )
{

	unsigned actualSample = firstPoint;
	for( int i = -CurveBasis::Degree; i < 0; ++i ) {
		for( unsigned j = 0; j < _lastSampleFrequency; ++j, ++actualSample ) {
			points[ actualSample ] = EvaluateACyclicCurve( i, values[ j ] );
		}
	}
	return actualSample;
}

template < typename VectorType >
unsigned
BSpline< VectorType >
::SampleUniformSplineACyclicEnd( unsigned firstPoint, PointSet< VectorType > &points, const typename BSpline< VectorType >::BFValVector &values )
{

	unsigned actualSample = firstPoint;
	for( int i = this->Size()-CurveBasis::Degree; i < (int)this->Size(); ++i ) {
		for( unsigned j = 0; j < _lastSampleFrequency; ++j, ++actualSample ) {
			points[ actualSample ] = EvaluateACyclicCurve( i, values[ j ] );
		}
	}
	points[ actualSample++ ] = EvaluateACyclicCurve( this->Size()-1, values[ 0 ] );
	return actualSample;
}

template < typename VectorType >
void
BSpline< VectorType >
::SampleWithDerivations( unsigned frequency )
{
	//TODO check
	_lastSampleFrequency = frequency;
	ReSampleWithDerivations();
}

template < typename VectorType >
void
BSpline< VectorType >
::ReSampleWithDerivations()
{
	ReSample();

	if( this->_pointCount <= 1 ) {
		return;
	}
	
	//Precompute basis functions values
	_lastBasisFunctionDerivationValues.resize( _lastSampleFrequency );
	double dt = 1.0 / _lastSampleFrequency;
	double t = 0.0;
	for( unsigned i=0; i < _lastSampleFrequency; ++i, t += dt ) {
		CurveBasis::DerivationsAtPoint( t, _lastBasisFunctionDerivationValues[ i ] );	
	}

	SampleWithFunctionValues( _lastSampleFrequency, _sampleDerivationCache, _lastBasisFunctionDerivationValues );

}

template < typename VectorType >
void
BSpline< VectorType >
::ResetSamples()
{

}

template < typename VectorType >
void
BSpline< VectorType >
::ResetSamplesDerivations()
{

}

template < typename VectorType >
typename BSpline< VectorType >::PointType
BSpline< VectorType >
::EvaluateCurve( int segment, const typename BSpline< VectorType >::BFunctionValues &values )
{
	PointType result;
	for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ segment + i ]);
	}
	return result;
	/*if( _cyclic ) {
		return EvaluateCyclicCurve( segment, values );
	} else {
		return EvaluateACyclicCurve( segment, values );
	}*/
}

template < typename VectorType >
typename BSpline< VectorType >::PointType
BSpline< VectorType >
::EvaluateCyclicCurve( int segment, const typename BSpline< VectorType >::BFunctionValues &values )
{ 
	PointType result;
	int count = this->_pointCount;
	for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			//TODO - correct
			int p = (segment + i);
			int idx = MOD( p, count );
			result += values[i]*(this->_points[ idx ]);
		}
	return result;
}

template < typename VectorType >
typename BSpline< VectorType >::PointType
BSpline< VectorType >
::EvaluateACyclicCurve( int segment, const BFunctionValues &values )
{ 
	PointType result;
	int count = this->_pointCount;

	if( segment < 0 ) {
		for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ max(segment+i,0) ]);
		}
		return result;
	} 
	if( segment >= (count-CurveBasis::Degree) ) {
		for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ min(segment+i,count-1) ]);
		}
		return result;
	} 

	for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ segment + i ]);
		}
	return result;
}

	
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*BSPLINE*/

