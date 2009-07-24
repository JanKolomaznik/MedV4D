#ifndef NEARESTNEIGHBOR_H_
#define NEARESTNEIGHBOR_H_

#include "base.h"

namespace M4D
{
namespace Imaging
{

template< typename ImageType >
class NearestNeighborInterpolator
	: public InterpolatorBase<ImageType>
{
public:
	typedef NearestNeighborInterpolator<ImageType> Self;
	typedef InterpolatorBase<ImageType> PredecessorType;
	typedef typename PredecessorType::CoordType CoordType;
	
	NearestNeighborInterpolator() 
		: PredecessorType() {}
	
	NearestNeighborInterpolator(const ImageType *image) 
		: PredecessorType(image) {}
	
	typename ImageType::Element Get(CoordType &coords);
};

}
}

//include implementation
#include "src/nearestNeighbour.tcc"

#endif /*NEARESTNEIGHBOR_H_*/
