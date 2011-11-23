/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ThresholdingFilter.h 
 * @{ 
 **/

#ifndef _THRESHOLDING_FILTER_H
#define _THRESHOLDING_FILTER_H

#include "common/Common.h"
#include "Imaging/ImageTraits.h"
#include "Imaging/AImageElementFilter.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

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
	: public AImageElementFilter< ImageType, ImageType, ThresholdingFunctor< typename ImageTraits< ImageType >::ElementType > >
{
public:
	typedef ThresholdingFunctor< typename ImageTraits< ImageType >::ElementType > 		Functor;
	typedef AImageElementFilter< ImageType, ImageType, Functor >			PredecessorType;
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

template< typename ElementType >
class ThresholdingMaskFunctor
{
public:
	void
	operator()( const ElementType&	input, uint8& output )
	{
		if( input < bottom || input > top ) {
			output = 0;
		} else {
			output = inValue;
		}
	}	

	ElementType	bottom;	
	ElementType	top;
		
	uint8			inValue;
};

template< typename ImageType >
class ThresholdingMaskFilter
	: public AImageElementFilter< ImageType, Image3DUnsigned8b, ThresholdingMaskFunctor< typename ImageTraits< ImageType >::ElementType > >
{
public:
	typedef ThresholdingMaskFunctor< typename ImageTraits< ImageType >::ElementType > 	Functor;
	typedef Imaging::AImageElementFilter< ImageType, Image3DUnsigned8b, Functor >	PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType 					InputElementType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): bottom( 0 ), top( 0 ), inValue( 255 ) {}

		InputElementType	bottom;	
		InputElementType	top;
		
		uint8			inValue;

		void
		CheckProperties() {
			_functor->bottom = bottom;
			_functor->top = top;
			_functor->inValue = inValue;
		}
		
		Functor	*_functor;
	};

	ThresholdingMaskFilter( Properties  * prop );
	ThresholdingMaskFilter();

	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Bottom, bottom );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, Top, top );
	GET_SET_PROPERTY_METHOD_MACRO( InputElementType, InValue, inValue );
protected:

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};
	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ThresholdingFilter.tcc"

#endif /*_THRESHOLDING_FILTER_H*/

/** @} */

