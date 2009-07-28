#ifndef NORMALIZED_MUTUAL_INFORMATION_H
#define NORMALIZED_MUTUAL_INFORMATION_H

#include <Imaging/criterion/CriterionBase.h>
#include <cmath>

namespace M4D
{

namespace Imaging
{

/**
 * Normalized Mutual Information criterion
 */
template< typename ElementType >
class NormalizedMutualInformation : public CriterionBase< ElementType >
{
public:

	/**
         * Calculate the normalized mutual information
         *  @param jointHist the joint histogram of the two images
	 *  @return the normalized mutual information
         */
	double compute( MultiHistogram< ElementType, 2 >& jointHist );

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "src/NormalizedMutualInformation.tcc"

#endif /*NORMALIZED_MUTUAL_INFORMATION_H*/
