#ifndef _THRESHOLDING_FILTER_H
#define _THRESHOLDING_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageElementFilter.h"


namespace M4D
{

namespace Imaging
{


template< typename ElementType >
class ThresholdingFunctor
{
public:
	void
	operator()( const ElementType&	input, ElementType& output )
	{
		if( input < bottom || input > top ) {
			output = outValue;
		} else {
			output = input;
		}
	}	

	ElementType	bottom;	
	ElementType	top;
		
	ElementType	outValue;
};

template< typename ImageType >
class ThresholdingFilter
	: public AbstractImageElementFilter< ImageType, ImageType, ThresholdingFunctor< typename ImageTraits< ImageType >::ElementType > >
{
public:
	typedef ThresholdingFunctor< typename ImageTraits< ImageType >::ElementType > 		Functor;
	typedef Imaging::AbstractImageElementFilter< ImageType, ImageType, Functor >		PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType 					InputElementType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): bottom( 0 ), top( 0 ), outValue( 0 ) {}

		InputElementType	bottom;	
		InputElementType	top;
		
		InputElementType	outValue;

		void
		CheckProperties() {
			_functor->bottom = bottom;
			_functor->top = top;
			_functor->outValue = outValue;
		}
		
		Functor	*_functor;
	};

	ThresholdingFilter( Properties  * prop );
	ThresholdingFilter();

	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Bottom, bottom );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Top, top );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, OutValue, outValue );
protected:

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

//******************************************************************************
//******************************************************************************
/*
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
*/
	
} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/filters/ThresholdingFilter.tcc"

#endif /*_THRESHOLDING_FILTER_H*/
