/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.h 
 * @{ 
 **/

#ifndef _SOBEL_EDGE_DETECTOR_H
#define _SOBEL_EDGE_DETECTOR_H

#include "common/Common.h"
#include "common/Types.h"
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
		Properties(): threshold( 50 ) {}

		ElementType	threshold;
	};

	SobelEdgeDetector( Properties * prop );
	SobelEdgeDetector();
	
	GET_SET_PROPERTY_METHOD_MACRO( ElementType, Threshold, threshold );
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

template< typename ImageType, typename OutImageType >
class SobelGradientOperator;

template< typename ImageType, typename OutType >
class SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >
	: public AbstractImage2DFilter< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >
{
public:	
	static const unsigned Dimension = ImageTraits< ImageType >::Dimension;
	typedef typename ImageTraits< ImageType >::ElementType 		ElementType;
	typedef SimpleVector< OutType, 2 >				OutElementType;
	typedef OutType							OutScalarType;
	typedef Image< OutElementType, ImageTraits< ImageType >::Dimension > OutImageType;
	typedef AbstractImage2DFilter< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > > 		PredecessorType;
	typedef ConvolutionMask<2,float32>				Mask;
	typedef typename ConvolutionMask<2,float32>::Ptr		MaskPtr;
	typedef ImageRegion< ElementType, 2 >				IRegion;
	typedef ImageRegion< OutElementType, 2 >			ORegion;

	struct Properties : public PredecessorType::Properties
	{
		Properties() {}
	};

	SobelGradientOperator( Properties * prop );
	SobelGradientOperator();
	
protected:
	bool
	Process2D(
			const IRegion	&inRegion,
			ORegion 	&outRegion
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

