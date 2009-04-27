#ifndef CRITERIONBASE_H
#define CRITERIONBASE_H

#include <Imaging/MultiHistogram.h>

namespace M4D
{

namespace Imaging
{

template< typename ElementType >
class CriterionBase
{
public:

	virtual double compute( MultiHistogram< ElementType, 2 >& jointHist, uint32 datasetSize ) = 0;

};

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*CRITERIONBASE_H*/
