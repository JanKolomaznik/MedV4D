#ifndef _ABSTRACT_IMAGE_ELEMENT_FILTER_H
#error File AbstractImageElementFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputElementType, typename OutputElementType, typename ElementFilter >
AbstractImageElementFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 >, ElementFilter >
::AbstractImageElementFilter( typename AbstractImageElementFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 >, ElementFilter >::Properties *prop ) 
	: PredecessorType( prop )
{
	
}

template< typename InputElementType, typename OutputElementType, typename ElementFilter >
bool
AbstractImageElementFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 >, ElementFilter >
::ProcessImage(
		const Image< InputElementType, 2 > 	&in,
		Image< OutputElementType, 2 >		&out
	    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	size_t width1;
	size_t height1;
	int32 xStride1;
	int32 yStride1;
	InputElementType *tmppointer1 = in.GetPointer( width1, height1, xStride1, yStride1 );
	InputElementType *pointer1 = NULL;

	size_t width2;
	size_t height2;
	int32 xStride2;
	int32 yStride2;
	OutputElementType *tmppointer2 = out.GetPointer( width2, height2, xStride2, yStride2 );
	OutputElementType *pointer2 = NULL;

	for( size_t j = 0; j < height1; ++j ) {
		pointer1 = tmppointer1;
		pointer2 = tmppointer2;
		for( size_t i = 0; i < width1; ++i ) {

			_elementFilter( *pointer1, *pointer2 );	

			pointer1 += xStride1;
			pointer2 += xStride2;
		}
		tmppointer1 += yStride1;
		tmppointer2 += yStride2;
	}
	return true;
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

template< typename InputElementType, typename OutputElementType, typename ElementFilter >
AbstractImageElementFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 >, ElementFilter >
::AbstractImageElementFilter( typename AbstractImageElementFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 >, ElementFilter >::Properties *prop ) 
	: PredecessorType( prop )
{
	
}

template< typename InputElementType, typename OutputElementType, typename ElementFilter >
bool
AbstractImageElementFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 >, ElementFilter >
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
	if( !this->CanContinue() ) {
		return false;
	}

	size_t width1;
	size_t height1;
	size_t depth1;
	int32 xStride1;
	int32 yStride1;
	int32 zStride1;
	InputElementType *tmppointer1 = in.GetPointer( width1, height1, depth1, xStride1, yStride1, zStride1 );
	InputElementType *pointer1 = NULL;

	size_t width2;
	size_t height2;
	size_t depth2;
	int32 xStride2;
	int32 yStride2;
	int32 zStride2;
	OutputElementType *tmppointer2 = out.GetPointer( width2, height2, depth2, xStride2, yStride2, zStride2 );
	OutputElementType *pointer2 = NULL;

	tmppointer1 += x1 * xStride1 + y1 * yStride1 + slice * zStride1;
	tmppointer2 += x1 * xStride2 + y1 * yStride2 + slice * zStride2;

	for( size_t j = y1; j < y2; ++j ) {
		pointer1 = tmppointer1;
		pointer2 = tmppointer2;
		for( size_t i = x1; i < x2; ++i ) {

			_elementFilter( *pointer1, *pointer2 );	

			pointer1 += xStride1;
			pointer2 += xStride2;
		}
		tmppointer1 += yStride1;
		tmppointer2 += yStride2;
	}
	return true;
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_ABSTRACT_IMAGE_ELEMENT_FILTER_H*/

