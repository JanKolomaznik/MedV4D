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

template< typename BufferType, typename SamplesSet, typename RegionType >
void
FillSampleValuesBufferFromRegion( BufferType &buffer, const SamplesSet & samples, RegionType &region )
{
	int32 sampleCount = samples.Size();
	buffer.resize( sampleCount );
	for( int32 i = 0; i < sampleCount; ++i ) {
		//TODO interpolation
		RasterPos pos = RasterPos( ROUND( samples[i][0] ), ROUND( samples[i][1] ) );
		buffer[ i ] = region.GetElement( pos );
	}
}

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

template< typename ContourType, typename FirstEnergyModel, typename SecondEnergyModel >
class DoubleEnergyFunctional
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::Type, ContourType::Dimension > 	GradientType;
	typedef Coordinates< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;
	typedef FirstEnergyModel	FirstEnergy;
	typedef SecondEnergyModel	SecondEnergy;
	
	DoubleEnergyFunctional(): _alpha( 1.0f ), _beta( 1.0f )
		{}

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		GradientType firstGradient;
		float32 firstGradientNorm = 0.0;
		firstGradient.Resize( gradient.Size() );

		GradientType secondGradient;
		float32 secondGradientNorm = 0.0;
		secondGradient.Resize( gradient.Size() );
		
		firstGradientNorm = _firstModel.GetParametersGradient( curve, firstGradient );
		if( Abs(firstGradientNorm) > Epsilon ) {
			firstGradientNorm = _alpha / firstGradientNorm;
		} else {
			firstGradientNorm = 0.0f;
		}

		secondGradientNorm = _secondModel.GetParametersGradient( curve, secondGradient );
		if( Abs(secondGradientNorm) > Epsilon ) {
			secondGradientNorm = _beta / secondGradientNorm;
		} else {
			secondGradientNorm = 0.0f;
		}

		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = (firstGradientNorm * firstGradient[i]) + (secondGradientNorm * secondGradient[i]);
		
			gradSize += gradient[i] * gradient[i];
		}
		return gradSize;
	}

	float32
	GetAlpha() const
		{ return _alpha; }

	void
	SetAlpha( float32 a )
		{ _alpha = a; }

	float32
	GetBeta() const
		{ return _beta; }

	void
	SetBeta( float32 b )
		{ _beta = b; }

	FirstEnergy &
	GetFirstModel()
		{ return _firstModel; }

	SecondEnergy &
	GetSecondModel()
		{ return _secondModel; }
private:
	FirstEnergyModel	_firstModel;
	SecondEnergyModel	_secondModel;

	float32			_alpha;
	float32			_beta;

};

template< typename ContourType >
class SimpleBaloonForce
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
		PointCoordinate	center;
		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			center += curve[i];
		}
		center *= 1.0f/static_cast<float32>( gradient.Size() );

		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = curve[i] - center;
			float32 size = sqrt(gradient[i]*gradient[i]);
			gradient[i] *= 1.0f/size;
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}
private:

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
		int32 sampleFrequency = curve.GetLastSampleFrequency();
		int32 sampleCount = samples.Size();
		static const int32 degree = ContourType::CurveBasis::Degree;
		//int32 segmentCount = curve.GetSegmentCount();
		
		//fill buffer with gradients on curve
		FillSampleValuesBufferFromRegion( _valBuffer, samples, _region );

		for( int32 i = 0; i < (int32)gradient.Size(); ++i ) {
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
	SetRegion( const RegionType &region )
		{ _region = region; }
private:
	RegionType		_region;
	ValuesAtSamplesBuffer	_valBuffer;
};

template< typename ContourType >
class UnifiedImageEnergy
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::Type, ContourType::Dimension > 	GradientType;
	typedef Coordinates< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;
	typedef typename RegionType::ElementType	ElementType;
	typedef std::vector< ElementType >		ValuesAtSamplesBuffer;

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}

		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = ComputePointGradient( i, curve );
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}
private:
	PointCoordinate
	ComputePointGradient( unsigned i, ContourType &curve )
	{
		PointCoordinate gradient = PointCoordinate( 0.0f );

		ComputeQfu( k, l, curve );
	}

	float32
	ComputeQfu( int32 k, int32 l, ContourType &curve )
	{
		return 0.0f;
	}

	RegionType		_region;
	ValuesAtSamplesBuffer	_valBuffer;
};

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*ENERGY_MODELS_H*/
