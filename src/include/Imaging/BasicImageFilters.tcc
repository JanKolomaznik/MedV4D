#ifndef _BASIC_IMAGE_FILTERS_H
#error File BasicImageFilters.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputElementType, typename OutputElementType >
CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::CopyImageFilter()
{

}

template< typename InputElementType, typename OutputElementType >
void
CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< OutputElementType, 3 >		&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    )
{
	for( size_t i = x1; i <= x2; ++i ) {
		for( size_t j = y1; j <= y2; ++j ) {
			out.GetElement( i, j, slice ) = in.GetElement( i, j, slice );
		}
	}
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_BASIC_IMAGE_FILTERS_H*/
