#ifndef OPTIMIZATION_BASE_H
#define OPTIMIZATION_BASE_H

#include "common/Vector.h"

namespace M4D
{
namespace Imaging
{

/**
 * Abstract base for optimization classes
 */
template< typename RegistrationFilterElementType, typename ElementType, uint32 dim >
class OptimizationBase
{

public:

	/**
	 * Optimize the given criterion function
	 *  @param v the input parameters
	 *  @param fret the return value
	 *  @param fil pointer to the registration filter that has the criterion function to optimize
	 */
	virtual void optimize(Vector< ElementType, dim > &v, ElementType &fret, RegistrationFilterElementType* fil ) = 0;

};

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*OPTIMIZATION_BASE_H*/
