/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImageElementFilter.h 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_ELEMENT_FILTER_H
#define _ABSTRACT_IMAGE_ELEMENT_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImage2DFilter.h"
#include <vector>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


/**
 * This template is prepared to ease design of image filters, which work on zero neighbourhood of element 
 * - use only value of the element.
 * These filters work with output dataset with same extents as input. 
 *
 * Because calling virtual method consumes time - this template uses different way of implementation of
 * actual computation - third parameter of template is functor which has implemented operator(), which takes 
 * two parameters - constant reference to input value, and reference to output value. This method is best to be inline and 
 * effective - its called on every element of input dataset.
 *
 * Specialization for processing 2D images.
 **/
template< typename InputImageType, typename OutputImageType, typename ElementFilter >
class AbstractImageElementFilter
	 : public AbstractImage2DFilter< InputImageType, OutputImageType >
{
public:
	typedef AbstractImage2DFilter< InputImageType, OutputImageType >	PredecessorType;
	typedef typename ImageTraits< InputImageType >::ElementType 		InputElementType;
	typedef typename ImageTraits< OutputImageType >::ElementType 		OutputElementType;
	
	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};
	
	AbstractImageElementFilter( Properties *prop );
	~AbstractImageElementFilter() {}

protected:

	bool
	Process2D(
			const ImageRegion< InputElementType, 2 > &inRegion,
			const ImageRegion< OutputElementType, 2 > &outRegion
		 );

	/*bool
	ProcessImage(
			const Image< InputElementType, 2 > 	&in,
			Image< OutputElementType, 2 >		&out
		    );*/


	ElementFilter	_elementFilter;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractImageElementFilter );
};

///**
// * This template is prepared to ease design of image filters, which work on zero neighbourhood of element 
// * - use only value of the element.
// * These filters work with output dataset with same extents as input. 
// *
// * Because calling virtual method consumes time - this template uses different way of implementation of
// * actual computation - third parameter of template is functor which has implemented operator(), which takes 
// * two parameters - constant reference to input value, and reference to output value. This method is best to be inline and 
// * effective - its called on every element of input dataset.
// *
// * Specialization for processing 3D images.
// **/
//template< typename InputElementType, typename OutputElementType, typename ElementFilter >
//class AbstractImageElementFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 >, ElementFilter >
//	 : public AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
//{
//public:
//	typedef typename Imaging::AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >	PredecessorType;
//	
//	struct Properties : public PredecessorType::Properties
//	{
//		Properties(): PredecessorType::Properties( 0, 10 ) {}
//
//	};
//	
//	AbstractImageElementFilter( Properties *prop );
//	~AbstractImageElementFilter() {}
//
//protected:
//
//	
//
//	bool
//	ProcessSlice(
//			const Image< InputElementType, 3 > 	&in,
//			Image< OutputElementType, 3 >		&out,
//			int32					x1,
//			int32					y1,
//			int32					x2,
//			int32					y2,
//			int32					slice
//		    );
//
//
//	ElementFilter	_elementFilter;
//private:
//	/**
//	 * Prohibition of copying.
//	 **/
//	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractImageElementFilter );
//};


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/AbstractImageElementFilter.tcc"

#endif /*_ABSTRACT_IMAGE_ELEMENT_FILTER_H*/

/** @} */

