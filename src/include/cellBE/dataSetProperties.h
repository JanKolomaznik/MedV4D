#ifndef DATASET_TYPES_H
#define DATASET_TYPES_H

#include "cellBE/iSerializable.h"

namespace M4D
{
namespace CellBE
{

enum DataSetType
{
  DATSET_IMAGE2D,
  DATSET_IMAGE3D,
  DATSET_IMAGE4D,
};

///////////////////////////////////////////////////////////////////////

class DataSetProperties : public iSerializable
{
};

///////////////////////////////////////////////////////////////////////

template<DataSetType dsetType>
class DataSetPropertiesTemplate : public DataSetProperties
{
  virtual uint8 GetType( void) = 0;
};

///////////////////////////////////////////////////////////////////////

struct Image2DProperties : public DataSetPropertiesTemplate<DATSET_IMAGE2D>
{
  uint16 x, y;

  uint8 GetType( void)
  {
    return (uint8)DATSET_IMAGE2D; 
  }
  void Serialize( NetStream &s)
  {
    s << x << y;
  }
  void DeSerialize( NetStream &s)
  {
    s >> x >> y;
  }
};

///////////////////////////////////////////////////////////////////////

struct Image3DProperties : public DataSetPropertiesTemplate<DATSET_IMAGE3D>
{
  uint16 x, y, z;
  uint8 GetType( void)
  {
    return (uint8)DATSET_IMAGE3D; 
  }
  void Serialize( NetStream &s)
  {
    s << x << y << z;
  }
  void DeSerialize( NetStream &s)
  {
    s >> x >> y >> z;
  }
};

///////////////////////////////////////////////////////////////////////

}}

#endif