/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file MedianSerializer.h 
 * @{ 
 **/

#ifndef MEDIAN_SERIALIZER_H
#define MEDIAN_SERIALIZER_H

#include "cellBE/AbstractFilterSerializer.h"
#include "Imaging/filters/MedianFilter.h"

namespace M4D
{
namespace CellBE
{

// supportig function
template< typename ElementType >
void
CreateMedianFilter( 
     M4D::Imaging::AbstractPipeFilter **resultingFilter
   , AbstractFilterSerializer **serializer
   , const uint16 id
   , M4D::CellBE::NetStream &s )
{
	typedef typename M4D::Imaging::Image< ElementType, 3 > ImageType;
	typedef typename M4D::Imaging::MedianFilter2D< ImageType > Filter;
  typedef typename FilterSerializer< Filter > FilterSerializer;

	Filter::Properties *prop = new Filter::Properties();

	*resultingFilter = new Filter( prop );
  *serializer = new FilterSerializer( prop, id);
}

/**
 *  MedianFilterSerializer.
 */
template< typename InputImageType >
class FilterSerializer< M4D::Imaging::MedianFilter2D< InputImageType > > 
	: public AbstractFilterSerializer
{
public:
	typedef typename M4D::Imaging::MedianFilter2D< InputImageType >
    ::Properties Properties;
	
	FilterSerializer( Properties * props, uint16 id) 
		: AbstractFilterSerializer( FID_Median, id )
		, _properties( props ) 
  {}

  void SerializeClassInfo( M4D::CellBE::NetStream &s)
  {
		s << (uint8) GetNumericTypeID< ImageTraits< InputImageType >::ElementType >();
  }

  void
  DeSerializeClassInfo( 
      M4D::Imaging::AbstractPipeFilter **resultingFilter
    , AbstractFilterSerializer **serializer
    , const uint16 id
    , M4D::CellBE::NetStream &s
    )
  {
		uint8 typeID;
		s >> typeID;
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( typeID, 
			CreateMedianFilter<TTYPE>( resultingFilter, serializer, id, s )
		);
  }

	void 
	SerializeProperties( M4D::CellBE::NetStream &s)
	{		
    s << (uint32)_properties->radius;
	}

	void
	DeSerializeProperties( M4D::CellBE::NetStream &s )
	{
		s >> (uint32)_properties->radius;
	}	
	
protected:
	Properties	*_properties;
};

}
}

#endif


/** @} */

