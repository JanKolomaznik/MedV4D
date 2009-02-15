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
	typedef Vector< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;

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
	typedef Vector< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;
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
	typedef Vector< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;

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
	typedef Vector< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;
	typedef typename RegionType::ElementType	ElementType;
	typedef typename ContourType::SamplePointSet	SamplePointSet;
	typedef std::vector< ElementType >		ValuesAtSamplesBuffer;

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

template< typename ContourType, typename RegionType1, typename RegionType2, typename Distribution >
class UnifiedImageEnergy
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::Type, ContourType::Dimension > 	GradientType;
	typedef Vector< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;
	typedef typename ContourType::BFValVector	BFValVector;
	typedef typename RegionType1::ElementType	ElementType1;
	typedef typename RegionType2::ElementType	ElementType2;
	typedef std::vector< float32 >			ValuesAtSamplesBuffer;
	typedef typename ContourType::SamplePointSet	SampleSet;
	static const unsigned Degree = ContourType::Degree;

	UnifiedImageEnergy() : _alpha(0.5f)
		{}

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}

		if( _sampleFrequency != (int32)curve.GetLastSampleFrequency() ) {
			RecalculateQki( curve );
		}

		FillSampleValuesBuffer( curve.GetSamplePoints() );		

		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = ComputePointGradient( i, curve );
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}

	void
	SetRegion1( const RegionType1 &region )
		{ _region1 = region; }
	void
	SetRegion2( const RegionType2 &region )
		{ _region2 = region; }

	float32
	GetAlpha() const
		{ return _alpha; }

	void
	SetAlpha( float32 a )
		{ _alpha = a; }
	
	Distribution &
	GetDistribution()
		{ return _distribution; }

private:
	float32
	ComputeValueAtPoint( const RasterPos &pos )
	{
		int value = _region1.GetElement( pos );
		/*
		float32 inProbability = _distribution.InProbability( value );// > 50 ? 0.1 : 0.9;
		float32 outProbability = _distribution.OutProbability( value );//value < 50 ? 0.15 : 0.85;
		float32 val1 = - log( inProbability / outProbability );
		*/
		float32 val1 = _distribution.LogProbabilityRatio( value );
		
		float32 val2 = _region2.GetElement( pos );
		
		return _alpha * val1 + (1-_alpha) * val2;
	}

	void
	FillSampleValuesBuffer( const SampleSet & samples )
	{
		int32 sampleCount = samples.Size();
		_valBuffer.resize( sampleCount );
		for( int32 i = 0; i < sampleCount; ++i ) {
			//TODO interpolation
			float32 x = samples[i][0];
			float32 y = samples[i][1];
			RasterPos pos = RasterPos( ROUND( x ), ROUND( y ) );
			_valBuffer[ i ] = ComputeValueAtPoint( pos );
		}
	}

	PointCoordinate
	ComputePointGradient( unsigned k, ContourType &curve )
	{
		PointCoordinate gradient = PointCoordinate( 0.0f );

		for( int32 i = k - Degree; i <= (int32)(k + Degree); ++i ) {
			gradient += curve.GetPointCyclic( i ) * ComputeIntegral( k, i, curve );
		}
		gradient = CoordinatesDimensionsShiftRight( gradient );
		gradient[0] *= -1;

		return gradient;
	}

	float32
	ComputeIntegral( int32 k, int32 i, ContourType &curve )
	{
		float32 result = 0.0f;
		int32 L = Max( k, i ) - Degree;
		int32 U = Min( k, i ) + 1;
		int32 sampleCount = curve.GetSamplePoints().Size();

		/*i = i < 0 ? i + curve.Size() : i;
		i = i >= (int32)curve.Size() ? i - curve.Size() : i;
		k = k < 0 ? k + curve.Size() : k;
		k = k >= (int32)curve.Size() ? k - curve.Size() : k;*/

		for( int32 j = L*_sampleFrequency; j < U*_sampleFrequency; ++j ) {

			int32 idx = MOD( j, sampleCount );
			//result += /*_valBuffer[ idx ]*/0.0f * Qki( k, i, idx );

			result += _valBuffer[ idx ] * Qki( k, i, j );
		}
		return result / (float32)_sampleFrequency;
	}
	
	float32
	Qki( int32 k, int32 i, int32 tR )
	{
		//int32 tLow = tR / _sampleFrequency;
		int32 tLow = floor( (float)tR / (float)_sampleFrequency );
		int32 nk = k - tLow;
		int32 ni = i - tLow;
		int32 t = tR - tLow * _sampleFrequency;
		
		if( nk > (int32)Degree ) { return 0.0f; }
		if( nk < 0 ) { return 0.0f; }
		if( ni > (int32)Degree ) { return 0.0f; }
		if( ni < 0 ) { return 0.0f; }
		if( t > (int32)_sampleFrequency ) { return 0.0f; }
		if( t < 0 ) { return 0.0f; }

		return Q[nk][ni][ t ];
	}

	void
	RecalculateQki( ContourType &curve )
	{
		const BFValVector &values = curve.GetLastBasisFunctionValues();
		const BFValVector &derivValues = curve.GetLastBasisFunctionDerivationValues();
		_sampleFrequency = curve.GetLastSampleFrequency();

		Q.resize( Degree+1 );
		for( int i = 0; i <= (int)Degree; ++i ) {
			Q[i].resize( Degree+1 );
			for( int j = 0; j <= (int)Degree; ++j ) {
				Q[i][j].resize( _sampleFrequency );
				for( int k = 0; k < _sampleFrequency; ++k ) {
					Q[i][j][k] = values[k][i] * derivValues[k][j];
				}
			}
		}
	}

	float32 		_alpha;

	RegionType1		_region1;
	RegionType2		_region2;
	Distribution		_distribution;
	ValuesAtSamplesBuffer	_valBuffer;

	std::vector< std::vector< std::vector< typename ContourType::Type > > > Q;
	
	int32 _sampleFrequency;
};

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*ENERGY_MODELS_H*/
