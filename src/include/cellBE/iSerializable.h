#ifndef ISERIALIZABLE_H
#define ISERIALIZABLE_H

#include "cellBE/netStream.h"

namespace M4D
{
namespace CellBE
{

class iSerializable
{
public:
  virtual void Serialize( NetStream &) = 0;
  virtual void DeSerialize( NetStream &s) = 0;
};

}
}

#endif

