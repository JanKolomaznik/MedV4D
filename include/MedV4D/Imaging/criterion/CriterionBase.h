#ifndef CRITERIONBASE_H
#define CRITERIONBASE_H

#include <Imaging/MultiHistogram.h>

namespace M4D
{

namespace Imaging
{

/**
 * Abstract base class for criterion implementations
 */
template< typename ElementType >
class CriterionBase
{
public:

	/**
	 * Calculate the criterion function
	 *  @param jointHist the joint histogram of the two images
	 *  @return the criterion value
	 */
	virtual double compute( MultiHistogram< ElementType, 2 >& jointHist ) = 0;

};

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*CRITERIONBASE_H*/
