#ifndef LINEAR_H_
#define LINEAR_H_

#include <cmath>

#include "base.h"
#include "Imaging/Image.h"

namespace M4D
{
namespace Imaging
{

template< typename ImageType >
class LinearInterpolator
	: public InterpolatorBase<ImageType>
{
public:
	typedef LinearInterpolator<ImageType> Self;
	typedef InterpolatorBase<ImageType> PredecessorType;
	typedef typename PredecessorType::CoordType CoordType;
	
	LinearInterpolator() 
		: PredecessorType() {}
	
	LinearInterpolator(const ImageType *image) 
		: PredecessorType(image) {}
	
	typename ImageType::Element Get(CoordType &coords);
};

}
}

//include implementation
#include "src/linear.tcc"

#endif /*LINEAR_H_*/
