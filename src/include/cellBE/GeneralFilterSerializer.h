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
  DeSerialize( M4D::CellBE::NetStream &s)
  {
	unsigned filterID;
	s >> filterID;
	return _serializers[ filterID ].DeSerializeProperties( s );
  }


  /**
   *  Returns pointer to filterSerializer based on given FilterProperties
   *  that represents a filter. Returned Serializer is later used for
   *  serializing the properties of the filter it represents.
   */
  template< typename FilterProperties >
  static AbstractFilterSerializer *
  GetFilterSerializer( FilterProperties *props );
	{
		return new FilterSerializer< FilterProperties >( props );
	}

};



//****************************************************************

//Empty declaration - we allow only partial specializations
template< typename FilterProperties >
class FilterSerializer;

template< typename ElementType, unsigned Dim >
M4D::Imaging::AbstractPipeFilter *
CreateThresholdingFilter( M4D::CellBE::NetStream &s )
{
	typedef typename M4D::Imaging::Image< ElementType, Dim > ImageType;
	typedef typename M4D::Imaging::ThresholdingFilter< ImageType > Filter;
	
	ElementType	bottom;	
	ElementType	top;
	ElementType	outValue;	

	Filter::Properties *prop = new Filter::Properties();

	s >> prop->bottom;
	s >> prop->top;
	s >> prop->outValue;

	return new Filter( prop );
}

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
	SerializeProperties( M4D::CellBE::NetStream &s)
	{
		s << this->GetID();

		s << ImageTraits< InputImageType >::Dimension;

		s << GetNumericTypeID< ImageTraits< InputImageType >::ElementType >;
		
		s << _properties->bottom;

		s << _properties->top;

		s << _properties->outValue;
	}

	M4D::Imaging::AbstractPipeFilter *
	DeSerializeProperties( M4D::CellBE::NetStream &s )
	{
		unsigned dim;
		unsigned typeID;
		
		s >> dim;

		s >> typeID;
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( typeID, 
			DIMENSION_TEMPLATE_SWITCH_MACRO( dim, return CreateThresholdingFilter<TTYPE, DIM >( s ) )
		);

	}
	
	
protected:
	Properties	*_properties;
};

typedef FilterSerializer< M4D::Imaging::ThresholdingFilter< Image3DUnsigned8b >::Properties ThresholdingSerializer;


}
}

#endif

