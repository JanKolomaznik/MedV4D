/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file GeneralFilterSerializer.h 
 * @{ 
 **/

#ifndef GENERAL_FILTER_SERIALIZER_H
#define GENERAL_FILTER_SERIALIZER_H

#include "FilterSerializerArray.h"

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
private:
  static FilterSerializerArray m_filterSerializers;

public:
  GeneralFilterSerializer();

  /**
   *  Read filterID from stream. Base on read filterID it instantiate
   *  appropriate FilterSerializer that performs actual deserialization
   *  and returns appropriate instance of filter
   */
  static void
  DeSerialize( M4D::Imaging::AbstractPipeFilter **resultingFilter
    , AbstractFilterSerializer **serializer
    , M4D::CellBE::NetStream &s)
  {
	  uint16 filterID;
	  s >> filterID;

    uint16 id;
    s >> id;

    m_filterSerializers.Get( (FilterID) filterID)->DeSerializeClassInfo( 
      resultingFilter, serializer, id, s);
  }

  /**
   *  Returns pointer to filterSerializer based on given FilterProperties
   *  that represents a filter. Returned Serializer is later used for
   *  serializing the properties of the filter it represents.
   */
	template< typename Filter >
	static AbstractFilterSerializer *
	GetFilterSerializer( typename Filter::Properties *props, uint16 id )
	{
		return new FilterSerializer< Filter >( props, id );
	}

};


}
}

#endif


/** @} */

