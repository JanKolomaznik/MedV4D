#ifndef PARAMETRIC_CURVE_H
#error File ParametricCurve.tcc cannot be included directly!
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
::BSpline()
{

}

template < typename CoordType, unsigned Dim >
BSpline< CoordType, Dim >
::BSpline( const PointSet< CoordType, Dim > & points )
{

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
	
	if( this->_pointCount <= 1 ) {
		return;
	}
	//Check if general
	int firstSegment = -CurveBasis::Degree+1;
	int lastSegment = this->_pointCount-1;
	if( _cyclic ) { 
		firstSegment = -1;
	}
	int segmentCount = lastSegment - firstSegment;
	int32 sampleCount = frequency * (segmentCount) + (_cyclic?0:1);

	//Ensure that we have enough space for samples
	_samplePointCache.Reserve( sampleCount );

	//Precompute basis functions values
	std::vector< BFunctionValues > precomputedFValues;
	precomputedFValues.reserve( frequency );
	double dt = 1.0 / frequency;
	double t = 0.0;
	for( unsigned i=0; i < frequency; ++i, t += dt ) {
		CurveBasis::ValuesAtPoint( t, precomputedFValues[ i ] );	
	}
	
	unsigned actualSample = 0;
	for( int i = firstSegment; i < lastSegment; ++i ) {
		for( unsigned j = 0; j < frequency; ++j, ++actualSample ) {
			_samplePointCache[ actualSample ] = EvaluateCurve( i, precomputedFValues[ j ] );
		}
	}
	if( !_cyclic ) {
		//Add last point
		_samplePointCache[ actualSample ] = EvaluateCurve( lastSegment, precomputedFValues[ 0 ] );
	}
}

template < typename CoordType, unsigned Dim >
void
BSpline< CoordType, Dim >
::SampleWithDerivations( unsigned frequency )
{

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

}

template < typename CoordType, unsigned Dim >
typename BSpline< CoordType, Dim >::PointType
BSpline< CoordType, Dim >
::EvaluateCurve( int segment, const BSpline< CoordType, Dim >::BFunctionValues &values )
{
	if( _cyclic ) {
		return EvaluateCyclicCurve( segment, values );
	} else {
		return EvaluateACyclicCurve( segment, values );
	}
}

template < typename CoordType, unsigned Dim >
typename BSpline< CoordType, Dim >::PointType
BSpline< CoordType, Dim >
::EvaluateCyclicCurve( int segment, const BSpline< CoordType, Dim >::BFunctionValues &values )
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
			result += values[i]*(this->_points[ MAX(segment+i,0) ]);
		}
		return result;
	} 
	if( segment >= (count-CurveBasis::Degree) ) {
		for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ MIN(segment+i,count-1) ]);
		}
		return result;
	} 

	for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ segment + i ]);
		}
	return result;
}

/*
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
	//Check if we have enough points
	if( this->_pointCount <= 1 ) {
		return;
	}
	int32 sampleCount = frequency * (this->_pointCount);
	if( !_cyclic ) { 
		sampleCount += frequency;
	}
	_samplePointCache.Reserve( sampleCount );

	std::vector< BFunctionValues > precomputedFValues;
	precomputedFValues.reserve( frequency );

	double dt = 1.0 / frequency;
	double t = 0.0;
	for( unsigned i=0; i < frequency; ++i, t += dt ) {
		CurveBasis::ValuesAtPoint( t, precomputedFValues[ i ] );	
	}
	
	unsigned actualSample = 0;
	//first segment - in cyclic version even last segment
	for( unsigned j = 0; j < frequency; ++j, ++actualSample ) {
		_samplePointCache[ actualSample ] = EvaluateCurve( -1, precomputedFValues[ j ] );
	}

	for( unsigned i = 0; i < this->_pointCount-1; ++i ) {
		for( unsigned j = 0; j < frequency; ++j, ++actualSample ) {
			_samplePointCache[ actualSample ] = EvaluateCurve( i, precomputedFValues[ j ] );
		}
	}
	if( !_cyclic ) {
		for( unsigned j = 0; j < frequency; ++j, ++actualSample ) {
			_samplePointCache[ actualSample ] = EvaluateCurve( this->_pointCount, precomputedFValues[ j ] );
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
	int count = this->_pointCount;
	for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			//result += values[i]*(this->_points[ (segment-curveEndSegCount) + i ]);
			int p = ((segment-curveEndSegCount) + i);
			int idx = MOD( p, count );
			result += values[i]*(this->_points[ idx ]);
		}
	return result;
}

template < typename CoordType, unsigned Dim, typename CurveBasis >
typename ParametricCurve< CoordType, Dim, CurveBasis >::PointType
ParametricCurve< CoordType, Dim, CurveBasis >
::EvaluateACyclicCurve( int segment, const BFunctionValues &values )
{ 
	PointType result;
	int count = this->_pointCount;

	if( segment < curveEndSegCount ) {
		for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			//TODO - correct
			int pom =  MAX(segment-curveEndSegCount+i,0);
			result += values[i]*(this->_points[ pom ]);
		}
		return result;
	} 
	if( segment >= (count-1-curveEndSegCount) ) {
		for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ MIN(segment-curveEndSegCount+i,count-1) ]);
		}
		return result;
	} 

	for( int i = 0; i <= CurveBasis::Degree; ++i ) {
			result += values[i]*(this->_points[ (segment-curveEndSegCount) + i ]);
		}
	return result;
}
*/
	
}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*PARAMETRIC_CURVE_H*/

