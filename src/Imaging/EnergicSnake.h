#ifndef ENERGIC_SNAKE_H
#define ENERGIC_SNAKE_H


namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file EnergicSnake.h 
 * @{ 
 **/

namespace Imaging
{
namespace Algorithms
{

/*class EnergyFunctional
{


};
*/

template< typename ContourType, typename EnergyModel >
class EnergicSnake
{
public:
	EnergicSnake();

	void
	Initialize( ContourType & contour );

	void
	Step();

	bool
	Converge();

	void
	Reset();

	EnergyModel&
	GetEnergyModel()
		{ return _energyFunctional; }
private:
	SwitchGradients();
	NormalizeGradient();

	EnergyModel	_energyFunctional;

	ContourType	_curve;

	GradientType	*_gradient;

	GradientType	*_gradients[2];
	unsigned	_actualGradient;

	float32		_stepScale;
};

template< typename ContourType, typename EnergyModel >
EnergicSnake::EnergicSnake()
{
	_gradients[0] = new GradientType;
	_gradients[1] = new GradientType;

	_actualGradient = 1;

	SwitchGradients();
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::Step()
{
	++step_count;

	ComputeCurveParametersGradient();

	UpdateCurveParameters();

	//Solve self intersection problem
	CheckSelfIntersection();

	//Divide or join segments with length out of tolerance
	CheckSegmentLengths();

}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::Converge()
{
	while( !Converged() ) {
		Step();
	}
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::ComputeCurveParametersGradient()
{
	(*_gradient).Reserve( _curve.Size() );

	_energyFunctional.GetParametersGradient( _curve, (*_gradient) );

	NormalizeGradient();
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::UpdateCurveParameters()
{
	ComputeStepScale();

	for( size_t i=0; i < _curve.Size(); ++i ) {
		_curve[i] += _stepScale * (*_gradient)[i];
	}
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::ComputeStepScale()
{
	_stepScale = 1.0;
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::CheckSelfIntersection()
{

}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::CheckSegmentLengths()
{

}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::SwitchGradients()
{
	_actualGradient = (_actualGradient + 1) % 2;

	_gradient = _gradients[_actualGradient];
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake::NormalizeGradient()
{
	double size = 0.0;
	for( size_t i=0; i < _gradient->Size(); ++i ) {
		size += (*_gradient)[i] * (*_gradient)[i];
	}

	size = 1.0 / sqrt( size );

	for( size_t i=0; i < _gradient->Size(); ++i ) {
		(*_gradient)[i] = size * (*_gradient)[i];
	}
}

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*ENERGIC_SNAKE_H*/
