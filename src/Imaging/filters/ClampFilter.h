#ifndef CLAMP_FILTER_H
#define CLAMP_FILTER_H

#include "common/Common.h"
#include "Imaging/ImageTraits.h"
#include "Imaging/AbstractImageElementFilter.h"
#include <cmath>

namespace M4D
{
namespace Imaging
{

template< typename ElementType >
class ClampingFunctor
{
public:
	void
	operator()( const ElementType&	input, ElementType& output )
	{
		if( input < bottom ) {
			output = bottom;
			return;
		} 
		if( input > top ) {
			output = top;
			return;
		}
		output = input;
	}	

	ElementType	bottom;	
	ElementType	top;
};

template< typename ImageType >
class ClampFilter :
	public AbstractImageElementFilter< ImageType, ImageType, ClampingFunctor< typename ImageTraits< ImageType >::ElementType > >
{
public:
	typedef ClampingFunctor< typename ImageTraits< ImageType >::ElementType > 	Functor;
	typedef AbstractImageElementFilter< ImageType, ImageType, Functor > 		PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType	ElementType;
	
	struct Properties : public PredecessorType::Properties
	{
	public:
		Properties(): bottom( TypeTraits< ElementType >::Min ), top( TypeTraits< ElementType >::Max ) {}

		ElementType	bottom;	
		ElementType	top;

		void
		CheckProperties() {
			_functor->bottom = bottom;
			_functor->top = top;
		}
		
		Functor	*_functor;

	};


	ClampFilter( Properties * prop ) :  PredecessorType( prop )
		{ this->_name = "ClampFilter"; GetProperties()._functor = &(this->_elementFilter); }
	ClampFilter() :  PredecessorType( new Properties() )
		{ this->_name = "ClampFilter"; GetProperties()._functor = &(this->_elementFilter); }

	GET_SET_PROPERTY_METHOD_MACRO( ElementType, Bottom, bottom );
	GET_SET_PROPERTY_METHOD_MACRO( ElementType, Top, top );
private:
	GET_PROPERTIES_DEFINITION_MACRO;
};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*CLAMP_FILTER_H*/
