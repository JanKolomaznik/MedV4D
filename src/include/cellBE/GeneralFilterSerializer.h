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
  static AbstractFilterSerializer *
  GetFilterSerializer(
    M4D::Imaging::AbstractPipeFilter::Properties *props);

};



}
}

#endif

