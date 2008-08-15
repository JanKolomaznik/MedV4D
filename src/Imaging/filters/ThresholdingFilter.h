#ifndef _THRESHOLDING_FILTER_H
#define _THRESHOLDING_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageSliceFilter.h"


namespace M4D
{

namespace Imaging
{


template< typename InputImageType >
class ThresholdingFilter;

template< typename InputElementType >
class ThresholdingFilter< Image< InputElementType, 2 > >
{
public:
	typedef AbstractPipeFilter  PredecessorType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): bottom( 0 ), top( 0 ), outValue( 0 ) {}

		InputElementType	bottom;	
		InputElementType	top;
		
		InputElementType	outValue;
	};

	ThresholdingFilter( Properties  * prop );
	ThresholdingFilter();

	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Bottom, bottom );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Top, top );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, OutValue, outValue );
protected:

	//TODO
private:
	GET_PROPERTIES_DEFINITION_MACRO;
};

template< typename InputElementType >
class ThresholdingFilter< Image< InputElementType, 3 > >
	: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:
	typedef typename  Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > > PredecessorType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): PredecessorType::Properties( 0, 10 ), bottom( 0 ), top( 0 ), outValue( 0 ) {}

		InputElementType	bottom;	
		InputElementType	top;
		
		InputElementType	outValue;
	};

	ThresholdingFilter( Properties  * prop );
	ThresholdingFilter();

	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Bottom, bottom );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Top, top );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, OutValue, outValue );
protected:

	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    );
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

//******************************************************************************
//******************************************************************************

template< typename InputImageType >
class ThresholdingFilterMask;

template< typename InputElementType >
class ThresholdingFilterMask< Image< InputElementType, 2 > >
{
	//TODO
};

template< typename InputElementType >
class ThresholdingFilterMask< Image< InputElementType, 3 > >
	: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image3DUnsigned8b >
{
public:
	typedef typename  Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image3DUnsigned8b > PredecessorType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): PredecessorType::Properties( 0, 10 ), bottom( 0 ), top( 0 ), inValue( 0 ), outValue( 0 ) {}

		InputElementType	bottom;	
		InputElementType	top;

		uint8			inValue;
		uint8			outValue;	

	};

	ThresholdingFilterMask();
protected:

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

private:
	GET_PROPERTIES_DEFINITION_MACRO;
};
	
} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/filters/ThresholdingFilter.tcc"

#endif /*_THRESHOLDING_FILTER_H*/
