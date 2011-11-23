#ifndef LINEAR_H_
#define LINEAR_H_

#include <cmath>

#include "base.h"
#include "Imaging/Image.h"

namespace M4D
{
namespace Imaging
{

/**
 * Linear interpolator
 */
template< typename ImageType >
class LinearInterpolator
	: public InterpolatorBase<ImageType>
{
public:
	typedef LinearInterpolator<ImageType> Self;
	typedef InterpolatorBase<ImageType> PredecessorType;
	typedef typename PredecessorType::CoordType CoordType;

	/**
	 * Constructor
	 */
	LinearInterpolator() 
		: PredecessorType() {}

	/**
	 * Constructor
	 *  @param image pointer to the image according to which interpolation is required
	 */
	LinearInterpolator(const ImageType *image) 
		: PredecessorType(image) {}

	/**
	 * Get the interpolated value
	 *  @param coord the coordinates where to calculate the interpolated value
	 *  @return the interpolated value
	 */
	typename ImageType::Element Get(CoordType &coords);
};

}
}

//include implementation
#include "src/linear.tcc"

#endif /*LINEAR_H_*/
