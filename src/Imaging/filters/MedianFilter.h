#ifndef _MEDIAN_FILTER_H
#define _MEDIAN_FILTER_H

#include "common/Common.h"
#include "Imaging/AbstractImage2DFilter.h"
#include <boost/shared_array.hpp>
#include <map>

#include "Imaging/FilterComputation.h"

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
	typedef ImageRegion< InputElementType, 2 >		Region;

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
			const Region	&inRegion,
			Region 		&outRegion
		 );


private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

//******************************************************************************
//******************************************************************************

} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/


//include implementation
#include "Imaging/filters/MedianFilter.tcc"

#endif /*_MEDIAN_FILTER_H*/


