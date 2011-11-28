#ifndef _MAX_FILTER_H
#define _MAX_FILTER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/AImage2DFilter.h"
#include <boost/shared_array.hpp>
#include <map>

#include "MedV4D/Imaging/FilterComputation.h"

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MaxFilter.h 
 * @{ 
 **/

namespace Imaging
{

template< typename InputImageType >
class MaxFilter2D
	: public AImage2DFilter< InputImageType, InputImageType >
{
public:	
	typedef AImage2DFilter< InputImageType, InputImageType > PredecessorType;
	typedef typename ImageTraits< InputImageType >::ElementType InputElementType;
	typedef ImageRegion< InputElementType, 2 >		Region;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): radius( 0 ) {}

		uint32	radius;
	};

	MaxFilter2D( Properties * prop );
	MaxFilter2D();

	GET_SET_PROPERTY_METHOD_MACRO( uint32, Radius, radius );
protected:

	void
	BeforeComputation( APipeFilter::UPDATE_TYPE &utype );

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
#include "MedV4D/Imaging/filters/MaxFilter.tcc"

#endif /*_MAX_FILTER_H*/


