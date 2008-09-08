/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file FilterSerializerArray.h 
 * @{ 
 **/

#ifndef FILTER_SERIALIZER_ARRAY_H
#define FILTER_SERIALIZER_ARRAY_H

#include "cellBE/AbstractFilterSerializer.h"
#include "Imaging/AbstractFilter.h"
#include "cellBE/filterSerializers/ThresholdingSerializer.h"
#include "cellBE/filterSerializers/MedianSerializer.h"
#include "cellBE/filterSerializers/SimpleMaxIntensityProjSerializer.h"

namespace M4D
{
namespace CellBE
{

/**
 *  This class is used to recognize particular FilterSerializers
 *  according typeID.
 */
class FilterSerializerArray
{
#define CURRENT_ARRAY_COUNT 3
private:
  AbstractFilterSerializer * m_serializerArray[CURRENT_ARRAY_COUNT];

public:
  FilterSerializerArray();

  AbstractFilterSerializer *Get( FilterID typeId) 
  {
    uint32 wantedIndex = (uint32) typeId; 
    if( wantedIndex < 0 || wantedIndex >= CURRENT_ARRAY_COUNT)
      throw WrongFilterException();

    return m_serializerArray[(uint32)typeId]; 
  }

};


}
}

#endif


/** @} */

