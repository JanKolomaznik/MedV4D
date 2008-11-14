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
	void
	Initialize( ContourType & contour );

	void
	Step();

	bool
	Converge();

	void
	Reset();
private:
	EnergyModel	_energyFunctional;

	ContourType	_curve;

	GradientType	_gradient;
};


Step()
{
	++step_count;

	ComputeCurveParametersGradient();

	UpdateCurveParameters();

	//Solve self intersection problem
	CheckSelfIntersection();

	//Divide or join segments with length out of tolerance
	CheckSegmentLengths();

}

Converge()
{
	while( !Converged() ) {
		Step();
	}
}

ComputeCurveParametersGradient()
{
	_gradient.Reserve( _curve.Size() );

	_energyFunctional.GetParametersGradient( _curve, _gradient );
}

UpdateCurveParameters()
{

}

CheckSelfIntersection()
{

}

CheckSegmentLengths()
{

}

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*ENERGIC_SNAKE_H*/
