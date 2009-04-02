/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Convolution.h 
 * @{ 
 **/

#ifndef FILTER_COMPUTATION_H
#define FILTER_COMPUTATION_H

#include "Imaging/Image.h"
#include "common/Vector.h"
#include "Imaging/ImageRegion.h"
#include <boost/shared_ptr.hpp>
#include "Imaging/ImageRegionAccessors.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< typename ElType >
class FilterFunctorBase
{
public:
	typedef ElType OutputValue;
};

template< typename Filter, typename Accessor >
class BasicFilterApplicator
{
public:
	//typedef Vector< int32, 2 > CoordType;
	typedef typename Accessor::CoordType CoordType;

	BasicFilterApplicator( Filter &filter, Accessor &accessor ) : _filter( filter ), _accessor( accessor )
	{ }

	void
	operator()( typename Filter::OutputValue &value, const CoordType &coordinates ) {
		value = _filter.Apply( coordinates, _accessor );
	}

	Filter		&_filter; 
	Accessor	&_accessor;
};

template< typename Filter, typename InputRegion, typename OutputRegion, template< typename Region > class Accessor  >
void
FilterProcessor( Filter &filter, const InputRegion &input, OutputRegion &output )
{
	typedef Accessor< InputRegion > AccessorType;
	AccessorType accessor( input );

	ForEachInRegion( output, BasicFilterApplicator< Filter, AccessorType >( filter, accessor ) );

}

template< typename Filter, typename InputRegion, typename OutputRegion, template< typename Region > class Accessor  >
void
FilterProcessorNeighborhood( Filter &filter, const InputRegion &input, OutputRegion &output )
{
	typedef Accessor< InputRegion > AccessorType;
	typedef SimpleAccessor< InputRegion > SimpleAccessorType;
	AccessorType accessor( input );
	SimpleAccessorType simpleAccessor( input );
	
	typename OutputRegion::PointType minimum = output.GetMinimum();
	typename OutputRegion::PointType maximum = output.GetMaximum();
	typename OutputRegion::PointType leftCorner = minimum - filter.GetLeftCorner();
	typename OutputRegion::PointType rightCorner = maximum - filter.GetRightCorner();
	typename OutputRegion::Iterator iterator = output.GetIterator( leftCorner, rightCorner );

	//TODO - solve borders
		
	ForEachByIterator( iterator, BasicFilterApplicator< Filter, SimpleAccessorType >( filter, simpleAccessor ) );

}

/*template< typename Filter, typename Region, template< typename Region > class Accessor  >
void
FilterProcessorInPlace( Filter &filter, OutputRegion &region )
{
	typedef Accessor< InputRegion > AccessorType;
	AccessorType accessor( region );

	ForEachInRegion( output, FilterApplicator< Filter, AccessorType >( filter, accessor ) );

}*/

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */
#endif /*FILTER_COMPUTATION_H*/

/** @} */

