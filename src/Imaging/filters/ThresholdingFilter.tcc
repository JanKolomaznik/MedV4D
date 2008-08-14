#ifndef _THRESHOLDING_FILTER_H
#error File ThresholdingFilter.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename InputElementType >
ThresholdingFilter< Image< InputElementType, 3 > >
::ThresholdingFilter() : PredecessorType( new Properties() )
{
	
}

template< typename InputElementType >
ThresholdingFilter< Image< InputElementType, 3 > >
::ThresholdingFilter( typename ThresholdingFilter< Image< InputElementType, 3 > >::Properties *prop ) : PredecessorType( prop )
{

}

template< typename InputElementType >
bool
ThresholdingFilter< Image< InputElementType, 3 > >
::ProcessSlice(	
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
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

	size_t width;
	size_t height;
	size_t depth;
	int32 xStride;
	int32 yStride;
	int32 zStride;
	InputElementType *pointer = in.GetPointer( width, height, depth, xStride, yStride, zStride );

	pointer += x1 * xStride + y1 * yStride + slice * zStride;
	for( size_t i = x1; i < x2; ++i ) {
		for( size_t j = y1; j < y2; ++j ) {
			if( GetProperties().bottom > *pointer || GetProperties().top < *pointer ) {
				*pointer = GetProperties().outValue;
			}
		}
	}
	/*for( size_t i = x1; i < x2; ++i ) {
		for( size_t j = y1; j < y2; ++j ) {
			InputElementType value = in.GetElement( i, j, slice );
			if( GetProperties().bottom <= value && GetProperties().top >= value ) {
				//unchanged
			} else {
				out.GetElement( i, j, slice ) = GetProperties().outValue;
			}
		}
	}*/
	return true;
}

//******************************************************************************
//******************************************************************************

template< typename InputElementType >
ThresholdingFilterMask< Image< InputElementType, 3 > >
::ThresholdingFilterMask() : PredecessorType( 0, 15 )
{

}

template< typename InputElementType >
bool
ThresholdingFilterMask< Image< InputElementType, 3 > >
::ProcessSlice(	
			const Image< InputElementType, 3 > 	&in,
			Image3DUnsigned8b			&out,
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
	for( size_t i = x1; i < x2; ++i ) {
		for( size_t j = y1; j < y2; ++j ) {
			InputElementType value = in.GetElement( i, j, slice );
			if( GetProperties().bottom <= value && GetProperties().top >= value ) {
				out.GetElement( i, j, value ) = GetProperties().inValue;
			} else {
				out.GetElement( i, j, value ) = GetProperties().outValue;
			}
		}
	}
	return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_THRESHOLDING_FILTER_H*/
