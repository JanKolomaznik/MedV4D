#ifndef _BASIC_IMAGE_FILTERS_H
#error File BasicImageFilters.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
CopyImageFilter< InputImageType, OutputImageType >::CopyImageFilter()
{

}

template< typename InputImageType, typename OutputImageType >
void
CopyImageFilter< InputImageType, OutputImageType >
::ProcessSlice(
			const InputImageType 	&in,
			OutputImageType		&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    )
{
	//TODO
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_BASIC_IMAGE_FILTERS_H*/
