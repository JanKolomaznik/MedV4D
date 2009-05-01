#ifndef OPTIMIZATION_BASE_H
#define OPTIMIZATION_BASE_H

#include "common/Vector.h"

namespace M4D
{
namespace Imaging
{

template< typename ElementType, uint32 dim >
class OptimizationBase
{

public:
	virtual void optimize(Vector< ElementType, dim > &v, ElementType &fret, ElementType func(Vector< ElementType, dim > &)) = 0;

};

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*OPTIMIZATION_BASE_H*/
