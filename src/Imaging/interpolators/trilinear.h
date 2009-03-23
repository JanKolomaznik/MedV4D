#ifndef TRILINEAR_H_
#define TRILINEAR_H_

#include <cmath>

#include "base.h"

namespace M4D
{
namespace Imaging
{

template< typename ImageType >
class TrilinearInterpolator
	: public InterpolatorBase<ImageType>
{
public:
	typedef TrilinearInterpolator<ImageType> Self;
	typedef InterpolatorBase<ImageType> PredecessorType;
	typedef typename PredecessorType::CoordType CoordType;
	
	TrilinearInterpolator(const ImageType *image) 
		: PredecessorType(image) {}
	
	typename ImageType::Element Get(CoordType &coords);
};

}
}

//include implementation
#include "src/trilinear.tcc"

#endif /*TRILINEAR_H_*/
