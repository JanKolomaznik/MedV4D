#ifndef NEARESTNEIGHBOR_H_
#define NEARESTNEIGHBOR_H_

#include <math.h>
#include "base.h"

namespace M4D
{
namespace Imaging
{

/**
 * Nearest Neighbor Interpolator
 */
template< typename ImageType >
class NearestNeighborInterpolator
	: public InterpolatorBase<ImageType>
{
public:
	typedef NearestNeighborInterpolator<ImageType> Self;
	typedef InterpolatorBase<ImageType> PredecessorType;
	typedef typename PredecessorType::CoordType CoordType;

	/**
         * Constructor
         */
	NearestNeighborInterpolator() 
		: PredecessorType() {}

	/**
         * Constructor
         *  @param image pointer to the image according to which interpolation is required
         */
	NearestNeighborInterpolator(const ImageType *image) 
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
#include "src/nearestNeighbour.tcc"

#endif /*NEARESTNEIGHBOR_H_*/
