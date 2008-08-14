#ifndef ABSTRACT_FILTER_SERIALIZER_H
#define ABSTRACT_FILTER_SERIALIZER_H

#include "filterIDEnums.h"
#include "cellBE/netStream.h"
#include "Common.h"

#include <vector>

namespace M4D
{
namespace CellBE
{

/**
 *  This base class defines interface for sucessing filterSerializers
 *  that have to implement it. Each filter template must have its own
 *  filterSerializer.
 */
class AbstractFilterSerializer
{
private:
  FilterID m_id;

public:
  AbstractFilterSerializer( FilterID id)
    : m_id( id)
  {
  }

  // Identification of particular AbstractFilter sucessor.
  FilterID GetID(void) { return m_id; }

  /**
   *  Each final sucessor has to implement this functions to allow
   *  sending all properties of that particular sucessor to server.
   */
  
  /**
   *  Each filter has its own set of attributes (we call them 
   *  filterProperties). SerializeProperties method has to put
   *  all the filterProperties to stream while DeSerializeProperties
   *  method has to retrieve them from the stream exactly in the same
   *  order that the SerializeProperties method has put them there.
   */
  virtual void SerializeProperties( M4D::CellBE::NetStream &s) = 0;
  virtual M4D::Imaging::AbstractPipeFilter *
    DeSerializeProperties( M4D::CellBE::NetStream &s) = 0;
};

/**
 *  Vector of FilterSettings. This vector is passed in param
 *  while creation of a new job. It defines actual remote pipeline
 *  that the created job represents.
 */
typedef std::vector<AbstractFilterSerializer *> FilterSerializerVector;


class WrongFilterException
  : public ExceptionBase
{
};

/**
 *  Empty declaration - we allow only partial specializations. These are
 *  in filterSerializers folder
 */
template< typename FilterType >
class FilterSerializer;

}
}

#endif

