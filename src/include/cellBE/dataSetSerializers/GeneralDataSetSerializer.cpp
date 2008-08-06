
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


void
GeneralDataSetSerializer::SerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j)
{
  // switch according dataType
}

///////////////////////////////////////////////////////////////////////////////