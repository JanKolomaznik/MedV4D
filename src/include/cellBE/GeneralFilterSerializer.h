#ifndef GENERAL_FILTER_SERIALIZER_H
#define GENERAL_FILTER_SERIALIZER_H

#include "AbstractFilterSerializer.h"
#include "Imaging/AbstractFilter.h"

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
  static M4D::Imaging::AbstractPipeFilter *
  DeSerialize( M4D::CellBE::NetStream &s)
  {
	  FilterID filterID;
	  s >> ((uint8 &) filterID);

    FilterSerializers::iterator it = m_filterSerializers.find( filterID);
    if( it != m_filterSerializers.end() )
      return it->second->DeSerializeProperties( s );
    else
      throw WrongFilterException();
  }

  /**
   *  Returns pointer to filterSerializer based on given FilterProperties
   *  that represents a filter. Returned Serializer is later used for
   *  serializing the properties of the filter it represents.
   */
  template< typename FilterProperties >
  static AbstractFilterSerializer *
  GetFilterSerializer( FilterProperties *props )
	{
		return new FilterSerializer< FilterProperties >( props );
	}

};


}
}

#endif

