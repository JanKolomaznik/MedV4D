#ifndef _THRESHOLDING_FILTER_H
#define _THRESHOLDING_FILTER_H

#include "Common.h"


namespace M4D
{

namespace Imaging
{

template< typename InputElementType >
struct ThresholdingOptions
{
	

};

//******************************************************************************
//******************************************************************************

template< typename InputElementType >
struct ThresholdingFilterMaskOptions
{
	InputElementType	bottom;	
	InputElementType	top;

	uint8			inValue;
	uint8			outValue;	

};

template< typename InputImageType >
class ThresholdingFilterMask;

template< typename InputElementType >
class ThresholdingFilterMask< Image< InputElementType, 2 >
{
	//TODO
};

template< typename InputElementType >
class ThresholdingFilterMask< Image< InputElementType, 3 > 
	: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image3DUnsigned8b >
{
public:
	typedef ThresholdingFilterMaskOptions< InputElementType >	Settings;
	ThresholdingFilterMask();
protected:
	typedef typename  Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image3DUnsigned8b > PredecessorType;

	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image3DUnsigned8b			&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    );

};
	
} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/ThresholdingFilter.tcc"

#endif /*_THRESHOLDING_FILTER_H*/
