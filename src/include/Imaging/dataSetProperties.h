#ifndef DATASET_PROPERTIES_ABSTRACT_H
#define DATASET_PROPERTIES_ABSTRACT_H

#include "../cellBE/iSerializable.h"
#include "dataSetTypeEnums.h"

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////

class DataSetPropertiesAbstract 
  : public M4D::CellBE::iSerializable
{
  DataSetType dataSetTypeID;

public:

  void SerializeIntoStream( M4D::CellBE::NetStream &stream)
  {
    stream << (uint8) dataSetTypeID;
    Serialize( stream);
  }  

  DataSetPropertiesAbstract( DataSetType dtype) : dataSetTypeID( dtype) {}
  inline DataSetType GetType( void) { return dataSetTypeID; }
};

///////////////////////////////////////////////////////////////////////

template<DataSetType dsetType>
class DataSetPropertiesTemplate : public DataSetPropertiesAbstract
{
  DataSetPropertiesTemplate() : DataSetPropertiesAbstract( dsetType) {}
};

///////////////////////////////////////////////////////////////////////

struct Image2DProperties : public DataSetPropertiesTemplate<DATSET_IMAGE2D>
{
  uint16 x, y;

  void Serialize( M4D::CellBE::NetStream &s)
  {
    s << x << y;
  }
  void DeSerialize( M4D::CellBE::NetStream &s)
  {
    s >> x >> y;
  }
};

///////////////////////////////////////////////////////////////////////

struct Image3DProperties : public DataSetPropertiesTemplate<DATSET_IMAGE3D>
{
  uint16 x, y, z;

  void Serialize( M4D::CellBE::NetStream &s)
  {
    s << x << y << z;
  }
  void DeSerialize( M4D::CellBE::NetStream &s)
  {
    s >> x >> y >> z;
  }
};

///////////////////////////////////////////////////////////////////////

}}

#endif