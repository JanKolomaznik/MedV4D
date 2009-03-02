#ifndef ENERGY_MODELS_2_H
#define ENERGY_MODELS_2_H

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file EnergyModels2.h 
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

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*ENERGY_MODELS_2_H*/
