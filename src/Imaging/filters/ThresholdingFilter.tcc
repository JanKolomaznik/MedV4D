#ifndef _THRESHOLDING_FILTER_H
#error File ThresholdingFilter.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename InputElementType >
ThresholdingFilterMask< Image< InputElementType, 3 > >
::ThresholdingFilter() : public PredecessorType( 0, 15 )
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
	Settings &settings = static_cast< Settings >( *_settings );

	for( size_t i = x1; i < x2; ++i ) {
		for( size_t j = y1; j < y2; ++j ) {
			InputElementType value = in.GetElement( i, j, slice );
			if( settings.bottom <= value && settings.top >= value ) {
				//unchanged
			} else {
				out.GetElement( i, j, value ) = settings.outValue;
			}
		}
	}
}

//******************************************************************************
//******************************************************************************

template< typename InputElementType >
ThresholdingFilterMask< Image< InputElementType, 3 > >
::ThresholdingFilterMask() : public PredecessorType( 0, 15 )
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
	Settings &settings = static_cast< Settings >( *_settings );

	for( size_t i = x1; i < x2; ++i ) {
		for( size_t j = y1; j < y2; ++j ) {
			InputElementType value = in.GetElement( i, j, slice );
			if( settings.bottom <= value && settings.top >= value ) {
				out.GetElement( i, j, value ) = settings.inValue;
			} else {
				out.GetElement( i, j, value ) = settings.outValue;
			}
		}
	}
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_THRESHOLDING_FILTER_H*/
