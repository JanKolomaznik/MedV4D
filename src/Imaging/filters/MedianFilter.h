#ifndef _MEDIAN_FILTER_H
#define _MEDIAN_FILTER_H

#include "common/Common.h"
#include "Imaging/AbstractImage2DFilter.h"
#include <boost/shared_array.hpp>
#include <map>

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MedianFilter.h 
 * @{ 
 **/

namespace Imaging
{

template< typename InputImageType >
class MedianFilter2D
	: public AbstractImage2DFilter< InputImageType, InputImageType >
{
public:	
	typedef AbstractImage2DFilter< InputImageType, InputImageType > PredecessorType;
	typedef typename ImageTraits< InputImageType >::ElementType InputElementType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): radius( 0 ) {}

		uint32	radius;
	};

	MedianFilter2D( Properties * prop );
	MedianFilter2D();

	GET_SET_PROPERTY_METHOD_MACRO( uint32, Radius, radius );
protected:

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	bool
	Process2D(
			const ImageRegion< InputElementType, 2 >	&inRegion,
			ImageRegion< InputElementType, 2 > 		&outRegion
		 );

	/*bool
	Process2D(
			InputElementType	*inPointer,
			int32			i_xStride,
			int32			i_yStride,
			InputElementType	*outPointer,
			int32			o_xStride,
			int32			o_yStride,
			uint32			width,
			uint32			height
		 );*/
private:
	typedef typename std::map< InputElementType, int >	Histogram;

	GET_PROPERTIES_DEFINITION_MACRO;

	inline InputElementType
	GetElementInOrder(
		Histogram				&histogram,
		uint32					order
	      );

};

//******************************************************************************
//******************************************************************************

} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/


//include implementation
#include "Imaging/filters/MedianFilter.tcc"

#endif /*_MEDIAN_FILTER_H*/


