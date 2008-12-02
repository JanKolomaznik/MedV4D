/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.h 
 * @{ 
 **/

#ifndef _SOBEL_EDGE_DETECTOR_H
#define _SOBEL_EDGE_DETECTOR_H

#include "Common.h"
#include "Types.h"
#include "Imaging/AbstractImage2DFilter.h"
#include "Imaging/Convolution.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename ImageType >
class SobelEdgeDetector 
	: public AbstractImage2DFilter< ImageType, ImageType >
{
public:	
	static const unsigned Dimension = ImageTraits< ImageType >::Dimension;
	typedef typename ImageTraits< ImageType >::ElementType 		ElementType;
	typedef AbstractImage2DFilter< ImageType, ImageType > 		PredecessorType;
	typedef ConvolutionMask<2,float32>				Mask;
	typedef typename ConvolutionMask<2,float32>::Ptr		MaskPtr;
	typedef ImageRegion< ElementType, 2 >				Region;

	struct Properties : public PredecessorType::Properties
	{
		Properties() {}
	};

	SobelEdgeDetector( Properties * prop );
	SobelEdgeDetector();
	
protected:
	bool
	Process2D(
			const Region	&inRegion,
			Region 		&outRegion
		 );
	MaskPtr		xMatrix;
	MaskPtr		yMatrix;

	void
	CreateMatrices();
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/SobelEdgeDetector.tcc"

#endif /*_SOBEL_EDGE_DETECTOR_H*/

/** @} */

