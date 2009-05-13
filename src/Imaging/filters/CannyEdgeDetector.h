/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file CannyEdgeDetector.h 
 * @{ 
 **/

#ifndef _CANNY_EDGE_DETECTOR_H
#define _CANNY_EDGE_DETECTOR_H

#include "common/Common.h"
#include "Imaging/AbstractImage2DFilter.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename ImageType >
class CannyEdgeDetector 
	: public AbstractImage2DFilter< ImageType, ImageType >
{
public:	
	static const unsigned Dimension = ImageTraits< ImageType >::Dimension;
	typedef typename ImageTraits< ImageType >::ElementType 		ElementType;
	typedef AbstractImage2DFilter< ImageType, ImageType > 		PredecessorType;
	typedef ImageRegion< ElementType, 2 >				Region;

	struct Properties : public PredecessorType::Properties
	{
		Properties(){}

	};

	CannyEdgeDetector( Properties * prop );
	CannyEdgeDetector();
	
	//GET_SET_PROPERTY_METHOD_MACRO( ElementType, Threshold, threshold );
protected:
	bool
	Process2D(
			const Region	&inRegion,
			Region 		&outRegion
		 );

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/CannyEdgeDetector.tcc"

#endif /*_CANNY_EDGE_DETECTOR_H*/

/** @} */

