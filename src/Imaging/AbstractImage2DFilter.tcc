/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImage2DFilter.tcc 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_2D_FILTER_H
#error File AbstractImage2DFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{
template< typename InputElementType, typename OutputElementType >
AbstractImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
::AbstractImage2DFilter( typename AbstractImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >::Properties *prop ) 
	: PredecessorType( prop )
{
	
}

template< typename InputElementType, typename OutputElementType >
bool
AbstractImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
::ProcessImage(
			const Image< InputElementType, 2 >	&in,
			Image< OutputElementType, 2 >		&out
		    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	return Process2D( in.GetRegion(), out.GetRegion() );

}

/*
template< typename InputElementType, typename OutputElementType >
bool
AbstractImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
::ProcessImage(
			const Image< InputElementType, 2 >	&in,
			Image< OutputElementType, 2 >		&out
		    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	uint32 width1;
	uint32 height1;
	int32 xStride1;
	int32 yStride1;
	InputElementType *pointer1 = in.GetPointer( width1, height1, xStride1, yStride1 );

	uint32 width2;
	uint32 height2;
	int32 xStride2;
	int32 yStride2;
	OutputElementType *pointer2 = out.GetPointer( width2, height2, xStride2, yStride2 );

	return Process2D( 
			pointer1, xStride1, yStride1, 
			pointer2, xStride2, yStride2, 
			width1, height1 );

}*/





template< typename InputElementType, typename OutputElementType >
AbstractImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::AbstractImage2DFilter( typename AbstractImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >::Properties *prop ) 
	: PredecessorType( prop )
{
	
}

template< typename InputElementType, typename OutputElementType >
bool
AbstractImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::ProcessSlice(
			const ImageRegion< InputElementType, 3 >	&inRegion,
			const ImageRegion< OutputElementType, 2 > 	&outRegion,
			int32						slice
	    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	return Process2D( inRegion.GetSlice( slice ), outRegion );

}

/*template< typename InputElementType, typename OutputElementType >
bool
AbstractImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::ProcessSlice(
		const Image< InputElementType, 3 > 	&in,
		Image< OutputElementType, 3 >		&out,
		int32					x1,
		int32					y1,
		int32					x2,
		int32					y2,
		int32					slice
	    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	uint32 width1;
	uint32 height1;
	uint32 depth1;
	int32 xStride1;
	int32 yStride1;
	int32 zStride1;
	InputElementType *pointer1 = in.GetPointer( width1, height1, depth1, xStride1, yStride1, zStride1 );

	uint32 width2;
	uint32 height2;
	uint32 depth2;
	int32 xStride2;
	int32 yStride2;
	int32 zStride2;
	OutputElementType *pointer2 = out.GetPointer( width2, height2, depth2, xStride2, yStride2, zStride2 );

	return Process2D( 
			pointer1 + slice * zStride1, xStride1, yStride1, 
			pointer2 + slice * zStride2, xStride2, yStride2, 
			width1, height1 );

}*/


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_ABSTRACT_IMAGE_2D_FILTER_H*/


/** @} */

