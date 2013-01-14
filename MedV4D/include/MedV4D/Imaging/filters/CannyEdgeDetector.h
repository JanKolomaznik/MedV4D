/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file CannyEdgeDetector.h 
 * @{ 
 **/

#ifndef _CANNY_EDGE_DETECTOR_H
#define _CANNY_EDGE_DETECTOR_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/AImage2DFilter.h"

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
	: public AImage2DFilter< ImageType, ImageType >
{
public:	
	static const size_t Dimension = ImageTraits< ImageType >::Dimension;
	typedef typename ImageTraits< ImageType >::ElementType 		ElementType;
	typedef AImage2DFilter< ImageType, ImageType > 		PredecessorType;
	typedef ImageRegion< ElementType, 2 >				Region;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): lowThreshold( 0.5 ), highThreshold( 0.5 ){}

		float32 lowThreshold;
		float32 highThreshold;
	};

	CannyEdgeDetector( Properties * prop );
	CannyEdgeDetector();
	
	GET_SET_PROPERTY_METHOD_MACRO( float32, LowThreshold, lowThreshold );
	GET_SET_PROPERTY_METHOD_MACRO( float32, HighThreshold, highThreshold );
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
#include "MedV4D/Imaging/filters/CannyEdgeDetector.tcc"

#endif /*_CANNY_EDGE_DETECTOR_H*/

/** @} */

