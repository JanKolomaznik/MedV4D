/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Convolution.h 
 * @{ 
 **/

#ifndef FILTER_COMPUTATION_H
#define FILTER_COMPUTATION_H

#include "Imaging/Image.h"
#include "Vector.h"
#include "Imaging/ImageRegion.h"
#include <boost/shared_ptr.hpp>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< unsigned Dim >
class MirrorAccessor
{
public:

};


template< typename Filter, typename InputRegion, typename OutputRegion, typename Accessor< unsigned dim > = MirrorAccessor  >
void
FilterProcessor( Filter &filter, const InputRegion &input, OutputRegion &output )
{
	Accessor< Input::Dimension > accessor( Input );

	filter.Init();

	ForEachInRegion( output, FilterApplicator( filter, accessor ) );

}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */
#endif /*FILTER_COMPUTATION_H*/

/** @} */

