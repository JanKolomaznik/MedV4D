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

};


Step()
{
	++step_count;

	ComputeCurveParametersGradient();

	UpdateCurveParameters();

	CheckSelfIntersection();

	CheckSegmentLengths();

}

Converge()
{
	while( !Converged() ) {
		Step();
	}
}

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*ENERGIC_SNAKE_H*/
