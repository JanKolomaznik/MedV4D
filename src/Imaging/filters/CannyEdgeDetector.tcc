/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file CannyEdgeDetector.tcc 
 * @{ 
 **/

#ifndef _CANNY_EDGE_DETECTOR_H
#error File CannyEdgeDetector.tcc cannot be included directly!
#else

#include "Imaging/CannyEdgeDetection.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ImageType >
CannyEdgeDetector< ImageType >
::CannyEdgeDetector() : PredecessorType( new Properties() )
{
}

template< typename ImageType >
CannyEdgeDetector< ImageType >
::CannyEdgeDetector( typename CannyEdgeDetector< ImageType >::Properties *prop ) 
: PredecessorType( prop ) 
{
}

template< typename ImageType >
bool
CannyEdgeDetector< ImageType >
::Process2D(
		const typename CannyEdgeDetector< ImageType >::Region	&inRegion,
		typename CannyEdgeDetector< ImageType >::Region 	&outRegion
		)
{
	Gradient<float32> * array = NULL;
	try {
		array = new Gradient<float32>[ inRegion.GetSize(0) * inRegion.GetSize(1) ];
		ImageRegion< Gradient<float32>, 2 > gradientRegion( array, inRegion.GetSize(), inRegion.GetMinimum() ); 

		CannyEdgeDetection( 
			inRegion,
			outRegion,
			gradientRegion
			);

		delete [] array;
	}
	catch( ... ) { 
		delete [] array;
		return false; 
	}

	return true;
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_CANNY_EDGE_DETECTOR_H*/

/** @} */

