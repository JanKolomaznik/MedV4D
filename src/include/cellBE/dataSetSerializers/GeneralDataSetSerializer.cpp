
#include "Common.h"
#include "cellBE/GeneralDataSetSerializer.h"

using namespace M4D::CellBE;
using namespace std;

///////////////////////////////////////////////////////////////////////////////

void
GeneralDataSetSerializer::SerializeDataSetProperties(
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::NetStream &s)
{
  /*switch( dataSet->GetType() )  TODO
  {
  case DataSetType::DATSET_IMAGE:

    break;

  case DataSetType::DATSET_TRIANGLE_MESH:
    break;
  }*/
}

///////////////////////////////////////////////////////////////////////////////

void
GeneralDataSetSerializer::SerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j)
{
  // switch according dataType
}

///////////////////////////////////////////////////////////////////////////////

void 
GeneralDataSetSerializer::DeSerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j)
{
}

///////////////////////////////////////////////////////////////////////////////

AbstractDataSetSerializer *
GeneralDataSetSerializer::GetDataSetSerializer( 
    M4D::Imaging::AbstractDataSet *dataSet)
{
  return NULL;
}

///////////////////////////////////////////////////////////////////////////////

M4D::Imaging::AbstractDataSet *
GeneralDataSetSerializer::DeSerializeDataSetProperties( 
      //AbstractDataSetSerializer **dataSetSerializer,
      M4D::CellBE::NetStream &s)
{
  return NULL;
}

///////////////////////////////////////////////////////////////////////////////