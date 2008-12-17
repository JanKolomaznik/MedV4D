#ifndef ENERGY_MODELS_H
#define ENERGY_MODELS_H

#include "Imaging/PointSet.h"
#include "Imaging/BSpline.h"
#include <vector>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file EnergyModels.h 
 * @{ 
 **/

namespace Imaging
{
namespace Algorithms
{

template< typename ContourType >
class EFConvergeToPoint
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::Type, ContourType::Dimension > 	GradientType;
	typedef Coordinates< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}
		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = _point - curve[i];
			float32 size = sqrt(gradient[i]*gradient[i]);
			float32 pom = (size - 100.0f)/size;
			gradient[i] = pom * gradient[i];
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}

	void
	SetCenterPoint( const PointCoordinate &point )
	{
		_point = point;
	}
private:
	PointCoordinate	_point;

};

template< typename ContourType, typename RegionType >
class GradientMagnitudeEnergy
{
public:
	typedef M4D::Imaging::Geometry::PointSet< typename ContourType::Type, ContourType::Dimension > 	GradientType;
	typedef Coordinates< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;
	typedef typename RegionType::ElementType	ElementType;
	typedef typename ContourType::SamplePointSet	SamplePointSet;
	typedef std::vector< ElementType >		ValuesAtSamplesBuffer;
	typedef Coordinates< int32, 2 >			RasterPos;

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}
		
		float32 gradSize = 0.0f;
		const SamplePointSet &samples = curve.GetSamplePoints();
		unsigned sampleFrequency = curve.GetLastSampleFrequency();
		int32 sampleCount = samples.Size();
		static const int32 degree = ContourType::Degree;
		int32 segmentCount = curve.GetSegmentCount();

		//fill buffer with gradients on curve
		_valBuffer.reserve( sampleCount );
		for( unsigned i = 0; i < sampleCount; ++i ) {
			//TODO interpolation
			_valBuffer[ i ] = _region.GetElement( RasterPos( ROUND( samples[i][0] ), ROUND( samples[i][1] ) ) );
		}

		for( int32 i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = PointCoordinate();
			
			for( int32 j = -degree; j < 1; ++j ) {
				for( int32 idx = 0; idx < sampleFrequency; ++idx ) {
					int pIdx = (i + j)*sampleFrequency + idx;
					if( pIdx < 0 ) pIdx += sampleCount;
					if( pIdx >= sampleCount ) pIdx -= sampleCount;

					//Get values for right t and point
					float coef = curve.GetLastBasisFunctionValues()[ idx ][ -1 * j ];
					gradient[i][0] += coef * _valBuffer[ pIdx ].data[0];
					gradient[i][1] += coef * _valBuffer[ pIdx ].data[1];
				}
			}
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}
	
	void
	SetRegion( RegionType &region )
		{ _region = region; }
private:
	RegionType		_region;
	ValuesAtSamplesBuffer	_valBuffer;
};


}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*ENERGY_MODELS_H*/
