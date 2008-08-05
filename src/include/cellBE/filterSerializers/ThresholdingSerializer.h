#ifndef THRESHOLDING_SERIALIZER_H
#define THRESHOLDING_SERIALIZER_H

#include "cellBE/AbstractFilterSerializer.h"

namespace M4D
{
namespace CellBE
{

class ThresholdingSerializer
  : public AbstractFilterSerializer
{
public:
  virtual DataSetType GetID(void) = 0;

  void SerializeProperties( M4D::CellBE::NetStream &s) {}
  void DeSerializeProperties( M4D::CellBE::NetStream &s) {}

  
};

}
}

#endif

