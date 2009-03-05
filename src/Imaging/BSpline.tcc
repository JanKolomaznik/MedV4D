#ifndef BSPLINE_H
#error File BSpline.tcc cannot be included directly!
#else

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


template < typename CoordType, unsigned Dim >
BSpline< CoordType, Dim >
::BSpline(): _cyclic( false ), _lastSampleFrequency( 2 )
{
	_samplePointCache.SetCyclic( _cyclic );
}

template < typename CoordType, unsigned Dim >
BSpline< CoordType, Dim >
::BSpline( const PointSet< CoordType, Dim > & points ): _cyclic( false ), _lastSampleFrequency( 2 )
{
	_samplePointCache.SetCyclic( _cyclic );
}

template < typename CoordType, unsigned Dim >
typename BSpline< CoordType, Dim >::PointType
BSpline< CoordType, Dim >
::PointByParameter( double t )const
{
	int segment_nmbr = floor( t );
	double relative_t = t - segment_nmbr;

	BFunctionValues values;

	CurveBasis::ValuesAtPoint( relative_t, values );

	return EvaluateCurve( segment_nmbr, values );
}

template < typename CoordType, unsigned Dim >
bool
BSpline< CoordType, Dim >
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

template < typename CoordType, unsigned Dim >
bool
BSpline< CoordType, Dim >
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

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::Sample( unsigned frequency )
{
	//TODO check
	_lastSampleFrequency = frequency;
	ReSample();
}

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::ReSample()
{
	if( this->_pointCount <= 1 ) {
		return;
	}
	
	//Precompute basis functions values
	_lastBasisFunctionValues.reserve( _lastSampleFrequency );
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

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::SampleWithFunctionValues( unsigned sampleFrequency, PointSet< CoordType, Dim > &points, const typename BSpline< CoordType, Dim >::BFValVector &values )
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

template < typename CoordType, unsigned Dim >
unsigned
BSpline< CoordType, Dim >
::SampleUniformSpline( unsigned firstPoint, PointSet< CoordType, Dim > &points, const typename BSpline< CoordType, Dim >::BFValVector &values )
{
	unsigned actualSample = firstPoint;
	for( int i = 0; i < (int)(this->Size()-CurveBasis::Degree); ++i ) {
		for( unsigned j = 0; j < _lastSampleFrequency; ++j, ++actualSample ) {
			points[ actualSample ] = EvaluateCurve( i, values[ j ] );
		}
	}
	return actualSample;
}

template < typename CoordType, unsigned Dim >
unsigned
BSpline< CoordType, Dim >
::SampleUniformSplineCyclicEnd( unsigned firstPoint, PointSet< CoordType, Dim > &points, const typename BSpline< CoordType, Dim >::BFValVector &values )
{
	unsigned actualSample = firstPoint;
	for( int i = 0; i < CurveBasis::Degree; ++i ) {
		for( unsigned j = 0; j < _lastSampleFrequency; ++j, ++actualSample ) {
			points[ actualSample ] = EvaluateCyclicCurve( i+this->Size()-CurveBasis::Degree, values[ j ] );
		}
	}
	return actualSample;
}

template < typename CoordType, unsigned Dim >
unsigned
BSpline< CoordType, Dim >
::SampleUniformSplineACyclicBegin( unsigned firstPoint, PointSet< CoordType, Dim > &points, const typename BSpline< CoordType, Dim >::BFValVector &values )
{

	unsigned actualSample = firstPoint;
	for( int i = -CurveBasis::Degree; i < 0; ++i ) {
		for( unsigned j = 0; j < _lastSampleFrequency; ++j, ++actualSample ) {
			points[ actualSample ] = EvaluateACyclicCurve( i, values[ j ] );
		}
	}
	return actualSample;
}

template < typename CoordType, unsigned Dim >
unsigned
BSpline< CoordType, Dim >
::SampleUniformSplineACyclicEnd( unsigned firstPoint, PointSet< CoordType, Dim > &points, const typename BSpline< CoordType, Dim >::BFValVector &values )
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

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::SampleWithDerivations( unsigned frequency )
{
	//TODO check
	_lastSampleFrequency = frequency;
	ReSampleWithDerivations();
}

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::ReSampleWithDerivations()
{
	ReSample();

	if( this->_pointCount <= 1 ) {
		return;
	}
	
	//Precompute basis functions values
	_lastBasisFunctionDerivationValues.reserve( _lastSampleFrequency );
	double dt = 1.0 / _lastSampleFrequency;
	double t = 0.0;
	for( unsigned i=0; i < _lastSampleFrequency; ++i, t += dt ) {
		CurveBasis::DerivationsAtPoint( t, _lastBasisFunctionDerivationValues[ i ] );	
	}

	SampleWithFunctionValues( _lastSampleFrequency, _sampleDerivationCache, _lastBasisFunctionDerivationValues );

}

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::ResetSamples()
{

}

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::ResetSamplesDerivations()
{

}

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::SplitSegment( int segment )
{
	int idx = segment + CurveBasis::HalfDegree;
	if( _cyclic ) {
		PointType newPoint = 0.5f * (this->_points[ MOD(idx, this->Size()) ]+this->_points[ MOD(idx+1, this->Size()) ]);
		this->InsertPoint( MOD(idx+1, this->Size()), newPoint );
	} else {
		_THROW_ ErrorHandling::ENotFinished( "SplitSegment() - Handling noncyclic splines" );
	}
}

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::JoinSegments( int segment )
{
	int idx = segment + (Degree+1 + 1)/2;
	this->RemovePoint( MOD(idx, this->Size()) );
	/*int idx = segment + CurveBasis::HalfDegree;
	if( _cyclic ) {
		PointType newPoint = 0.5f * (this->_points[ MOD(idx, this->Size()) ]+this->_points[ MOD(idx+1, this->Size()) ]);
		this->_points[ MOD(idx+1, this->Size()) ] = newPoint;
		this->RemovePoint( MOD(idx, this->Size()) );
	} else {
		_THROW_ ErrorHandling::ENotFinished( "SplitSegment() - Handling noncyclic splines" );
	}*/
}

template < typename CoordType, unsigned Dim >
typename BSpline< CoordType, Dim >::PointType
BSpline< CoordType, Dim >
::EvaluateCurve( int segment, const typename BSpline< CoordType, Dim >::BFunctionValues &values )
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

template < typename CoordType, unsigned Dim >
typename BSpline< CoordType, Dim >::PointType
BSpline< CoordType, Dim >
::EvaluateCyclicCurve( int segment, const typename BSpline< CoordType, Dim >::BFunctionValues &values )
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

template < typename CoordType, unsigned Dim >
typename BSpline< CoordType, Dim >::PointType
BSpline< CoordType, Dim >
::EvaluateACyclicCurve( int segment, const BFunctionValues &values )
{ 
	PointType result;
	int count = this->_pointCount;

	if( segment < 0 ) {
		for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ Max(segment+i,0) ]);
		}
		return result;
	} 
	if( segment >= (count-CurveBasis::Degree) ) {
		for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ Min(segment+i,count-1) ]);
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

