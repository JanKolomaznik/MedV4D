#ifndef THRESHOLDING_SERIALIZER_H
#define THRESHOLDING_SERIALIZER_H

#include "cellBE/AbstractFilterSerializer.h"
#include "Imaging/filters/ThresholdingFilter.h"

namespace M4D
{
namespace CellBE
{

// supportig function
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

/**
 *  ThresholdingFilterSerializer.
 */
template< typename InputImageType >
class FilterSerializer< typename M4D::Imaging::ThresholdingFilter< InputImageType >::Properties > 
	: public AbstractFilterSerializer
{
public:
	typedef typename M4D::Imaging::ThresholdingFilter< InputImageType >::Properties Properties;
	
	FilterSerializer( Properties * props) 
		: AbstractFilterSerializer( GetFilterID( *props ) ), _properties( props ) 
  {}

	void 
	SerializeProperties( M4D::CellBE::NetStream &s)
	{
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

}
}

#endif

