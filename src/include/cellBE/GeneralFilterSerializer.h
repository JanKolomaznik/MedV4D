#ifndef GENERAL_FILTER_SERIALIZER_H
#define GENERAL_FILTER_SERIALIZER_H

#include "AbstractFilterSerializer.h"
#include "Imaging/AbstractFilter.h"
#include "cellBE/filterSerializers/ThresholdingSerializer.h"

#include <map>

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
  typedef std::map<FilterID, AbstractFilterSerializer *> FilterSerializers;
  static FilterSerializers m_filterSerializers;

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
	  FilterID filterID;
	  s >> ((uint8 &) filterID);

    uint16 id;
    s >> id;

    FilterSerializers::iterator it = m_filterSerializers.find( filterID);
    if( it != m_filterSerializers.end() )
      it->second->DeSerializeClassInfo( resultingFilter, serializer, id, s);
    else
      throw WrongFilterException();
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

