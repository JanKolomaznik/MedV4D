/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImage2DFilter.tcc 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_2D_FILTER_H
#error File AImage2DFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{
template< typename InputElementType, typename OutputElementType >
AImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
::AImage2DFilter( typename AImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >::Properties *prop ) 
	: PredecessorType( prop )
{
	
}

template< typename InputElementType, typename OutputElementType >
bool
AImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
::ProcessImage(
			const Image< InputElementType, 2 >	&in,
			Image< OutputElementType, 2 >		&out
		    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	ImageRegion< OutputElementType, 2 > outRegion = out.GetRegion();
	return Process2D( in.GetRegion(), outRegion );

}

/*
template< typename InputElementType, typename OutputElementType >
bool
AImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
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
AImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::AImage2DFilter( typename AImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >::Properties *prop ) 
	: PredecessorType( prop )
{
	
}

template< typename InputElementType, typename OutputElementType >
bool
AImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::ProcessSlice(
			const ImageRegion< InputElementType, 3 >	&inRegion,
			ImageRegion< OutputElementType, 2 > 		&outRegion,
			int32						slice
	    )
{
	if( !this->CanContinue() ) {
		return false;
	}

	return Process2D( inRegion.GetSliceRel( slice ), outRegion );

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_ABSTRACT_IMAGE_2D_FILTER_H*/


/** @} */

