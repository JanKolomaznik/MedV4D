#ifndef _THRESHOLDING_FILTER_H
#error File ThresholdingFilter.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

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

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_THRESHOLDING_FILTER_H*/
