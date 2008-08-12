#ifndef GENERAL_FILTER_SERIALIZER_H
#define GENERAL_FILTER_SERIALIZER_H

#include "AbstractFilterSerializer.h"
#include "Imaging/AbstractFilter.h"

namespace M4D
{
namespace CellBE
{

/**
 *  This class is used to recognize particular FilterSerializers
 *  according typeID.
 */
class GeneralFilterSerializer
{
public:
  /**
   *  Read filterID from stream. Base on read filterID it instantiate
   *  appropriate FilterSerializer that performs actual deserialization
   *  and returns appropriate instance of filter
   */
  static M4D::Imaging::AbstractPipeFilter *
  DeSerialize( M4D::CellBE::NetStream &s);

  /**
   *  Returns pointer to filterSerializer based on given FilterProperties
   *  that represents a filter. Returned Serializer is later used for
   *  serializing the properties of the filter it represents.
   */
  template< typename FilterProperties >
  static AbstractFilterSerializer *
  GetFilterSerializer( FilterProperties *props );

};


template< typename FilterProperties >
AbstractFilterSerializer *
GeneralFilterSerializer::GetFilterSerializer( FilterProperties *props )
{
	return new FilterSerializer< FilterProperties >( props );
}


//****************************************************************

//Empty declaration - we allow only partial specializations
template< typename FilterProperties >
class FilterSerializer;

template< typename InputImageType >
class FilterSerializer< M4D::Imaging::ThresholdingFilter< InputImageType >::Properties > 
	: public AbstractFilterSerializer
{
public:
	typedef typename M4D::Imaging::ThresholdingFilter< InputImageType >::Properties Properties;
	
	FilterSerializer( Properties * props) 
		: AbstractFilterSerializer( GetFilterID( *props ) ), _properties( props ) {}

	~FilterSerializer() { delete _properties; }

	void 
	SerializeProperties( M4D::CellBE::NetStream &s);
	
protected:
	Properties	_properties;
};


}
}

#endif

