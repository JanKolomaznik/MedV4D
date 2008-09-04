/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file ThresholdingSerializer.h 
 * @{ 
 **/

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
void
CreateThresholdingFilter( 
     M4D::Imaging::AbstractPipeFilter **resultingFilter
   , AbstractFilterSerializer **serializer
   , const uint16 id
   , M4D::CellBE::NetStream &s )
{
	typedef M4D::Imaging::Image< ElementType, Dim > ImageType;
	typedef M4D::Imaging::ThresholdingFilter< ImageType > Filter;
	typedef FilterSerializer< Filter > FilterSerializer;

	typename Filter::Properties *prop = new typename Filter::Properties();

	*resultingFilter = new Filter( prop );  // id
  *serializer = new FilterSerializer( prop, id);  // id
}

/**
 *  ThresholdingFilterSerializer.
 */
template< typename InputImageType >
class FilterSerializer< M4D::Imaging::ThresholdingFilter< InputImageType > > 
	: public AbstractFilterSerializer
{
public:
	typedef typename M4D::Imaging::ThresholdingFilter< InputImageType >::Properties Properties;
	
	FilterSerializer( Properties * props, uint16 id) 
		: AbstractFilterSerializer( FID_Thresholding, id )
		, _properties( props ) 
  {}

  void SerializeClassInfo( M4D::CellBE::NetStream &s)
  {
    s << (uint8) M4D::Imaging::ImageTraits< InputImageType >::Dimension;
		s << (uint8) GetNumericTypeID< typename M4D::Imaging::ImageTraits< InputImageType >::ElementType >();
  }

  void
  DeSerializeClassInfo( 
      M4D::Imaging::AbstractPipeFilter **resultingFilter
    , AbstractFilterSerializer **serializer
    , const uint16 id
    , M4D::CellBE::NetStream &s
    )
  {
    uint8 dim;
		uint8 typeID;
		
		s >> dim;
		s >> typeID;
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( typeID, 
			DIMENSION_TEMPLATE_SWITCH_MACRO( 
        dim, CreateThresholdingFilter<TTYPE, DIM >( 
          resultingFilter, serializer, id, s ) )
		);
  }

	void 
	SerializeProperties( M4D::CellBE::NetStream &s)
	{		
		s << _properties->bottom << _properties->top << _properties->outValue;
	}

	void
	DeSerializeProperties( M4D::CellBE::NetStream &s )
	{
		s >> _properties->bottom >> _properties->top >> _properties->outValue;
	}	
	
protected:
	Properties	*_properties;
};

}
}

#endif


/** @} */

