/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.tcc 
 * @{ 
 **/

#ifndef _CONVOLUTION_FILTER_H
#error File ConvolutionFilter.tcc cannot be included directly!
#else

#include "Imaging/FilterComputation.h"
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
	this->_name = "ConvolutionFilter2D";
}

template< typename ImageType, typename MatrixElement >
ConvolutionFilter2D< ImageType, MatrixElement >
::ConvolutionFilter2D( typename ConvolutionFilter2D< ImageType, MatrixElement >::Properties *prop ) 
: PredecessorType( prop ) 
{
	this->_name = "ConvolutionFilter2D";
}

template< typename ImageType, typename MatrixElement >
bool
ConvolutionFilter2D< ImageType, MatrixElement >
::Process2D(
		const typename ConvolutionFilter2D< ImageType, MatrixElement >::Region	&inRegion,
		typename ConvolutionFilter2D< ImageType, MatrixElement >::Region 	&outRegion
		)
{
	try {
		ConvolutionFilterFtor< ElementType > filter( *GetConvolutionMask() );
		FilterProcessorNeighborhood< 
			ConvolutionFilterFtor< ElementType >,
			Region,
			Region,
			MirrorAccessor
			>( filter, inRegion, outRegion );	
	}
	catch( ... ) { 
		return false; 
	}

	return true;
}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_CONVOLUTION_FILTER_H*/

/** @} */

