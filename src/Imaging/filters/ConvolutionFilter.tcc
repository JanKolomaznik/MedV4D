/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.tcc 
 * @{ 
 **/

#ifndef _CONVOLUTION_FILTER_H
#error File ConvolutionFilter.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ImageType, typename MatrixElement >
ConvolutionFilter2D< ImageType, MatrixElement >
::ConvolutionFilter2D() : PredecessorType( new Properties() )
{

}

template< typename ImageType, typename MatrixElement >
ConvolutionFilter2D< ImageType, MatrixElement >
::ConvolutionFilter2D( ConvolutionFilter2D< ImageType, MatrixElement >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< typename ImageType, typename MatrixElement >
bool
ConvolutionFilter2D< ImageType, MatrixElement >
::Process2D(
		const ConvolutionFilter2D< ImageType, MatrixElement >::Region	&inRegion,
		ConvolutionFilter2D< ImageType, MatrixElement >::Region 	&outRegion
		)
{
	try {
		Compute2DConvolution( 
				inRegion, 
				outRegion, 
				*(GetProperties().matrix), 
				GetProperties().addition, 
				GetProperties().multiplication 
				);
	}
	catch( ... ) { 
		return false; 
	}

	return true;
}

//******************************************************************************
//******************************************************************************

/*
template< typename InputImageType, typename MatrixElement >
ConvolutionFilter3D< InputImageType, MatrixElement >::Properties
::Properties() : PredecessorType::Properties( 0, 10 ), width( 1 ), height( 1 ), depth( 1 )
{
	matrix = MatrixPtr( new MatrixElement[1] );

	matrix[0] = 1;
}

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
			int32			x1,	
			int32			y1,	
			int32			x2,	
			int32			y2,	
			int32			slice
		    )
{

}*/

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_CONVOLUTION_FILTER_H*/

/** @} */

