/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.h 
 * @{ 
 **/

#ifndef _CONVOLUTION_FILTER_H
#define _CONVOLUTION_FILTER_H

#include "common/Common.h"
#include "Imaging/AImage2DFilter.h"
#include <boost/shared_array.hpp>
#include "Imaging/Convolution.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename ImageType, typename MatrixElement = float32 >
class ConvolutionFilter2D 
	: public AImage2DFilter< ImageType, ImageType >
{
public:	
	static const unsigned Dimension = ImageTraits< ImageType >::Dimension;
	typedef typename ImageTraits< ImageType >::ElementType ElementType;
	typedef AImage2DFilter< ImageType, ImageType > PredecessorType;
	typedef typename ConvolutionMask<2,MatrixElement>::Ptr	MaskPtr;
	typedef ImageRegion< ElementType, 2 >		Region;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): multiplication( TypeTraits< MatrixElement >::One ), 
		addition( TypeTraits< ElementType >::Zero ) {}

		MaskPtr 	matrix; 
		MatrixElement 	multiplication;
		ElementType	addition;
	};

	ConvolutionFilter2D( Properties * prop );
	ConvolutionFilter2D();
	
	GET_SET_PROPERTY_METHOD_MACRO( MaskPtr, ConvolutionMask, matrix );
	GET_SET_PROPERTY_METHOD_MACRO( MatrixElement, Multiplication, multiplication );
	GET_SET_PROPERTY_METHOD_MACRO( ElementType, Addition, addition );
protected:
	bool
	Process2D(
			const Region	&inRegion,
			Region 		&outRegion
		 );
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ConvolutionFilter.tcc"

#endif /*_CONVOLUTION_FILTER_H*/

/** @} */

