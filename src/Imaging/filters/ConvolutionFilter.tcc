#ifndef _THRESHOLDING_FILTER_H
#error File ConvolutionFilter.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename InputElementType, typename MatrixElement >
ConvolutionFilter2D< Image< InputElementType, 3 >, MatrixElement >
::ConvolutionFilter2D() : PredecessorType( new Properties() )
{

}

template< typename InputElementType, typename MatrixElement >
ConvolutionFilter2D< Image< InputElementType, 3 >, MatrixElement >
::ConvolutionFilter2D( ConvolutionFilter2D< Image< InputElementType, 3 >, MatrixElement >::Properties *prop ) 
: PredecessorType( prop ), 
{

}

template< typename InputElementType, typename MatrixElement >
bool
ConvolutionFilter2D< Image< InputElementType, 3 >, MatrixElement >
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
	int hwidth = settings.width / 2;
	int hheight = settings.height / 2;

	//TODO check
	for( size_t i = x1 + hwidth; i < ( x2 - settings.width + hwidth ); ++i ) {
		for( size_t j = y1 + hheight; j < ( y2 - settings.height + hheight ); ++j ) {
			MatrixElement tmp = 0.0;	
			for( size_t ii = 0; ii < settings.width; ++ii ) {
				for( size_t jj = 0; jj < settings.height; ++jj ) {
					tmp += settings.matrix[ settings.width * jj + ii ] 
						* in.GetElement( i + ii - hwidth, j + jj - hheight, slice );
				}
			}
			out.GetElement( i, j, value ) = static_cast<InputElementType>( tmp );
		}
	}
}

//******************************************************************************
//******************************************************************************


template< typename InputElementType >
ConvolutionFilter3D< Image< InputElementType, 3 > >
::ConvolutionFilter3D() : public PredecessorType( 0, 15 )
{

}

template< typename InputElementType >
bool
ConvolutionFilter3D< Image< InputElementType, 3 > >
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

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_THRESHOLDING_FILTER_H*/
