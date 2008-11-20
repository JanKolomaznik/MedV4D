/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImageElementFilter.tcc 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_ELEMENT_FILTER_H
#error File AbstractImageElementFilter.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< typename InputImageType, typename OutputImageType, typename ElementFilter >
AbstractImageElementFilter< InputImageType, OutputImageType, ElementFilter >
::AbstractImageElementFilter( typename AbstractImageElementFilter< InputImageType, OutputImageType, ElementFilter >::Properties *prop ) 
	: PredecessorType( prop )
{
	
}

template< typename InputImageType, typename OutputImageType, typename ElementFilter >
bool
AbstractImageElementFilter< InputImageType, OutputImageType, ElementFilter >
::Process2D(
		const ImageRegion< typename AbstractImageElementFilter< InputImageType, OutputImageType, ElementFilter >::InputElementType, 2 > &inRegion,
		const ImageRegion< typename AbstractImageElementFilter< InputImageType, OutputImageType, ElementFilter >::OutputElementType, 2 > &outRegion
	    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	ImageIterator< InputElementType, 2 > iIterator = inRegion.GetIterator();
	ImageIterator< OutputElementType, 2 > oIterator = outRegion.GetIterator();

	while( !iIterator.IsEnd() ) {
	
		_elementFilter( *iIterator, *oIterator );

		++iIterator;
		++oIterator;
	}


	return true;
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_IMAGE_ELEMENT_FILTER_H*/


/** @} */

