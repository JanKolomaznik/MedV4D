#ifndef NORMALIZED_MUTUAL_INFORMATION_H
#define NORMALIZED_MUTUAL_INFORMATION_H

#include <Imaging/criterion/CriterionBase.h>
#include <cmath>

namespace M4D
{

namespace Imaging
{

template< typename ElementType >
class NormalizedMutualInformation : public CriterionBase< ElementType >
{
public:

	double compute( MultiHistogram< ElementType, 2 >& jointHist, uint32 datasetSize );

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "src/NormalizedMutualInformation.tcc"

#endif /*NORMALIZED_MUTUAL_INFORMATION_H*/
