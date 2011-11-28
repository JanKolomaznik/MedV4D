#ifndef _MASK_MEDIAN_FILTER_H
#define _MASK_MEDIAN_FILTER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/AImage2DFilter.h"

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MaskMedianFilter.h 
 * @{ 
 **/

namespace Imaging
{

template< unsigned Dim >
class MaskMedianFilter2D
	: public AImage2DFilter< Image< uint8, Dim >, Image< uint8, Dim > >
{
public:	
	typedef AImage2DFilter< Image< uint8, Dim >, Image< uint8, Dim > > 	PredecessorType;
	typedef uint8									ElementType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): radius( 0 ) {}

		uint32	radius;
	};

	MaskMedianFilter2D( Properties * prop );
	MaskMedianFilter2D();

	GET_SET_PROPERTY_METHOD_MACRO( uint32, Radius, radius );
protected:

	void
	BeforeComputation( APipeFilter::UPDATE_TYPE &utype );

	bool
	Process2D(
			const ImageRegion< uint8, 2 >	&inRegion,
			ImageRegion< uint8, 2 > 	&outRegion
		 );
	/*bool
	Process2D(
			ElementType	*inPointer,
			int32		i_xStride,
			int32		i_yStride,
			ElementType	*outPointer,
			int32		o_xStride,
			int32		o_yStride,
			uint32		width,
			uint32		height
		 );*/
private:
	static const uint8 TRUE_VALUE = 255;
	
	struct Histogram
	{
		int32 &
		operator[]( bool counter )
		{
			if( counter ) {
				return trueCounter;
			}
			return falseCounter;
		}

		void
		clear() 
		{
			trueCounter = falseCounter = 0;
		}

		int32 trueCounter;
		int32 falseCounter;
	};

	GET_PROPERTIES_DEFINITION_MACRO;

	inline ElementType
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
#include "MedV4D/Imaging/filters/MaskMedianFilter.tcc"

#endif /*_MASK_MEDIAN_FILTER_H*/


